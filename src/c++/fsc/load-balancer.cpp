#include "load-balancer.h"
#include "data.h"

#include <random>

namespace fsc {

namespace {

struct Backend {
	virtual ~Backend() noexcept(false) {};
	virtual Maybe<capnp::Capability::Client> getTarget(uint16_t methodId) = 0;
	virtual void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) = 0;
};

struct BackendImpl : public Backend {
	LoadBalancerConfig::Reader globalConfig;
	LoadBalancerConfig::Backend::Reader config;
	
	NetworkInterface::Client networkInterface;
	
	Promise<void> maintenanceTask;
	
	bool errorLogged = false;
	Maybe<capnp::Capability::Client> target = nullptr;
	
			
	BackendImpl(LoadBalancerConfig::Reader gConf, NetworkInterface::Client ni, LoadBalancerConfig::Backend::Reader lConf) :
		globalConfig(gConf), config(lConf),
		networkInterface(mv(ni)),
		
		maintenanceTask(NEVER_DONE)
	{
		if(!config.hasName()) {
			KJ_LOG(WARNING, "Backend has no name, status message will show URL instead", config.getUrl());
		}
		maintenanceNow();
	}
	
	void maintenanceNow() {
		maintenanceTask = performMaintenance().eagerlyEvaluate([this](kj::Exception error) {
			KJ_LOG(ERROR, "Maintenance task for backend failed", config.getUrl(), error);
		});
	}
	
	Promise<void> retryAfter(kj::Duration time) {
		return getActiveThread().timer().afterDelay(time)
		.then([this]() { return performMaintenance(); });
	}
	
	Promise<void> performMaintenance() {
		KJ_IF_MAYBE(pTarget, target) {
			// Perform a heartbeat request to a bogus interface
			auto heartbeat = pTarget -> typelessRequest(0, 0, nullptr, capnp::Capability::Client::CallHints());
			return heartbeat.send().ignoreResult()
			.catch_([this](kj::Exception&& e) {
				switch(e.getType()) {
					// This is what we expect
					case kj::Exception::Type::UNIMPLEMENTED:
						return;
					
					default:
						kj::throwFatalException(mv(e));
				}
			})
			.then(
				// Connection is OK
				[this]() {
					return retryAfter(globalConfig.getHeartbeatIntervalSeconds() * kj::SECONDS);
				},
				
				// Heartbeat call failed
				[this](kj::Exception error) {
					KJ_LOG(WARNING, "Lost connection to backend, reconnecting ...", config.getUrl(), error);
					target = nullptr;
					return performMaintenance();
				}
			);
			
		} else {
			// Try to connect to target
			
			// Run connection
			auto connectRequest = networkInterface.connectRequest();
			connectRequest.setUrl(config.getUrl());
			
			return connectRequest.send()
			.then(
				[this](auto response) {
					errorLogged = false;
					
					auto conn = response.getConnection();
					
					target = conn.getRemoteRequest().sendForPipeline().getRemote();
					
					KJ_LOG(INFO, "Connected to backend", config.getUrl());
					
					return performMaintenance();
				},
				[this](kj::Exception error) {
					if(!errorLogged) {
						KJ_LOG(WARNING, "Failed to connect to backend", config.getUrl(), error);
						errorLogged = true;
					} else {
						KJ_LOG(INFO, "Failed to connect to backend", config.getUrl(), error);
					}

					return retryAfter(globalConfig.getReconnectIntervalSeconds() * kj::SECONDS);
				}
			);
		}
	}
	
	Maybe<capnp::Capability::Client> getTarget(uint16_t) override {
		return target;
	}
	
	void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) override {
		using B = LoadBalancer::StatusInfo::Backend;
		B result;
		result.url = kj::heapString(config.hasName() ? config.getName() : config.getUrl());
		
		if(target != nullptr) {
			result.status = B::OK;
		} else {
			result.status = B::DISCONNECTED;
		}
		
		out.add(mv(result));
	}
};

struct Pool : public Backend {
	kj::Array<BackendImpl> backends = nullptr;

	std::mt19937 prng;
	
	uint32_t randomOffset = 0;
	uint32_t counter = 0;
	
	Pool(LoadBalancerConfig::Reader globalConfig, NetworkInterface::Client ni, capnp::List<LoadBalancerConfig::Backend>::Reader configs) {
		auto backendsBuilder = kj::heapArrayBuilder<BackendImpl>(configs.size());
		for(auto c : configs)
			backendsBuilder.add(globalConfig, ni, c);
		
		backends = backendsBuilder.finish();
	}
	
	Maybe<capnp::Capability::Client> getTarget(uint16_t methodId) {
		for(auto retries : kj::indices(backends)) {				
			// Adjust counter
			if(counter > backends.size()) {
				counter = 0;
				randomOffset = prng();
			}
			
			uint32_t index = counter + randomOffset;
			++counter;
			
			index %= backends.size();				
			auto maybeResult = backends[index].getTarget(methodId);
			
			KJ_IF_MAYBE(pResult, maybeResult) {
				return *pResult;
			} else {
				// Move on to next offset
			}
		}
		
		KJ_FAIL_REQUIRE("No pool backends reachable");
	}
	
	void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) override {
		for(auto& b : backends)
			b.buildStatus(out);
	}
};

struct Rule : public Backend {
	LoadBalancerConfig::Rule::Reader ruleConfig;
	
	Pool pool;
	
	Rule(LoadBalancerConfig::Reader globalConfig, NetworkInterface::Client ni, LoadBalancerConfig::Rule::Reader config) :
		ruleConfig(config),
		pool(globalConfig, mv(ni), config.getPool())
	{}
	
	Maybe<capnp::Capability::Client> getTarget(uint16_t methodId) override {
		auto matches = ruleConfig.getMatches();
		switch(matches.which()) {
			case LoadBalancerConfig::Rule::Matches::ALL:
				break;
			
			case LoadBalancerConfig::Rule::Matches::ANY_OF: {
				bool false = true;
				for(auto candidate : matches.getAnyOf()) {
					if(methodId == candidate)
						found = true;
				}
				
				if(!found)
					return nullptr;
				break;
			}
			
			case LoadBalancerConfig::Rule::Matches::ALL_EXCEPT: {
				for(auto candidate : matches.getAllExcept()) {
					if(methodId == candidate)
						return nullptr;
				}
				break;
			}
		}
		
		return pool.getTarget(methodId);
	}
	
	void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) override {
		pool.buildStatus(out);
	}
};

struct RuleSet : public Backend {
	kj::Array<Rule> rules;
	
	RuleSet(LoadBalancerConfig::Reader config, NetworkInterface::Client ni) {
		auto rulesBuilder = kj::heapArrayBuilder<Rule>(config.getRules().size());
		for(auto ruleConfig : config.getRules()) {
			rulesBuilder.add(config, ni, ruleConfig);
		}
		rules = rulesBuilder.finish();
	}
	
	Maybe<capnp::Capability::Client> getTarget(uint16_t methodId) override {
		for(auto& rule : rules) {
			KJ_IF_MAYBE(pResult, rule.getTarget(methodId)) {
				return *pResult;
			}
		}
		
		return nullptr;
	}
	
	void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) override {
		for(auto& r : rules)
			r.buildStatus(out);
	}
};
	
struct LoadBalancerImpl : public capnp::Capability::Server, public kj::Refcounted, public LoadBalancer {	
	NetworkInterface::Client networkInterface;
	Temporary<LoadBalancerConfig> config;
	
	RuleSet ruleSet;
	
	LoadBalancerImpl(NetworkInterface::Client networkInterface_, LoadBalancerConfig::Reader config_) :
		networkInterface(mv(networkInterface_)), config(config_),
		ruleSet(config, networkInterface)
	{}
		
	Own<LoadBalancer> addRef() override { return kj::addRef(*this); }
	
	StatusInfo status() override {
		kj::Vector<StatusInfo::Backend> backends;
		ruleSet.buildStatus(backends);
		
		StatusInfo result;
		result.backends = backends.releaseAsArray();
		return result;
	}
	
	capnp::Capability::Client loadBalanced() override {
		return Own<capnp::Capability::Server>(kj::addRef(*this));
	}
	
	capnp::Capability::Client selectCallTarget(uint16_t methodId) {
		KJ_IF_MAYBE(pTarget, ruleSet.getTarget(methodId)) {
			return *pTarget;
		}
		
		KJ_FAIL_REQUIRE("No matching load-balancing rule for method", methodId);
	}
	
	DispatchCallResult dispatchCall(
		uint64_t interfaceId, uint16_t methodId,
        capnp::CallContext<capnp::AnyPointer, capnp::AnyPointer> context
	) override {
		// TODO: Message size?
		auto tailRequest = selectCallTarget(methodId)
			.typelessRequest(interfaceId, methodId, nullptr, capnp::Capability::Client::CallHints());
		
		return DispatchCallResult {
			context.tailCall(mv(tailRequest)),
			false, // isStreaming
			true   // allowCancellation
		};
	}
};

}

Own<LoadBalancer> newLoadBalancer(NetworkInterface::Client clt, LoadBalancerConfig::Reader config) {
	return kj::refcounted<LoadBalancerImpl>(mv(clt), config);
}

}