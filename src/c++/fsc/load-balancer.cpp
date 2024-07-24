#include "load-balancer.h"
#include "data.h"

#include <random>

namespace fsc {

namespace {

struct Backend {
	virtual ~Backend() noexcept(false) {};
	virtual Maybe<capnp::Capability::Client> getTarget(uint64_t schemaId, uint16_t methodId) = 0;
	virtual void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) = 0;
};

struct BackendImpl : public Backend {
	LoadBalancerConfig::Reader globalConfig;
	LoadBalancerConfig::Backend::Reader config;
	
	NetworkInterface::Client networkInterface;
	
	Promise<void> maintenanceTask;
	
	bool errorLogged = false;
	Maybe<capnp::Capability::Client> target = nullptr;
	bool ok = false;
	
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
		if(ok) {
			// Perform a heartbeat request to a bogus interface
			auto heartbeat = connect().typelessRequest(0, 0, nullptr, capnp::Capability::Client::CallHints());
			return heartbeat.send().ignoreResult()
			.catch_([this](kj::Exception&& e) -> Promise<void> {
				switch(e.getType()){
					// This is what we expect
					case kj::Exception::Type::UNIMPLEMENTED:
						return READY_NOW;
					
					default:
						return mv(e);
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
					ok = false;
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
					ok = true;
					
					auto conn = response.getConnection();
					
					if(config.getPersistent())
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
	
	capnp::Capability::Client connect() {
		KJ_IF_MAYBE(pTarget, target) {
			return *pTarget;
		}
		
		auto connectRequest = networkInterface.connectRequest();
		connectRequest.setUrl(config.getUrl());
		connectRequest.setAllowCompression(config.getCompressed());
		
		return connectRequest.sendForPipeline().getConnection().getRemoteRequest().sendForPipeline().getRemote();
	}
	
	Maybe<capnp::Capability::Client> getTarget(uint64_t schemaId, uint16_t methodId) override {
		if(ok)
			return connect();
		
		return nullptr;
	}
	
	void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) override {
		using B = LoadBalancer::StatusInfo::Backend;
		B result;
		result.url = kj::heapString(config.hasName() ? config.getName() : config.getUrl());
		
		if(ok) {
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
	
	Maybe<capnp::Capability::Client> getTarget(uint64_t schemaId, uint16_t methodId) override {
		for(auto retries : kj::indices(backends)) {				
			// Adjust counter
			if(counter > backends.size()) {
				counter = 0;
				randomOffset = prng();
			}
			
			uint32_t index = counter + randomOffset;
			++counter;
			
			index %= backends.size();				
			auto maybeResult = backends[index].getTarget(schemaId, methodId);
			
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
	
	Own<Backend> impl;
	
	Rule(LoadBalancerConfig::Reader globalConfig, NetworkInterface::Client ni, LoadBalancerConfig::Rule::Reader rule) :
		ruleConfig(rule)
	{
		if(rule.isPool()) {
			impl = kj::heap<Pool>(globalConfig, mv(ni), rule.getPool());
		} else if(rule.isBackend()) {
			impl = kj::heap<BackendImpl>(globalConfig, mv(ni), rule.getBackend());
		} else {
			KJ_FAIL_REQUIRE("Unknown rule type", rule);
		}
	}
	
	Maybe<capnp::Capability::Client> getTarget(uint64_t schemaId, uint16_t methodId) override {
		auto check = [&](LoadBalancerConfig::Rule::MethodSpec::Reader spec) {
			if(spec.getInterface() != 0 && spec.getInterface() != schemaId)
				return false;
			
			for(auto m : spec.getMethods()) {
				if(m == methodId) return true;
			}
			
			return false;
		};
		
		auto matches = ruleConfig.getMatches();
		switch(matches.which()) {
			case LoadBalancerConfig::Rule::Matches::ALL:
				goto match;

			case LoadBalancerConfig::Rule::Matches::ONLY:
				if(check(matches.getOnly()))
					goto match;
				goto no_match;
			
			case LoadBalancerConfig::Rule::Matches::ANY_OF: {
				for(auto spec : matches.getAnyOf()) {
					if(check(spec))
						goto match;
				}
				goto no_match;
			}
			
			case LoadBalancerConfig::Rule::Matches::ALL_EXCEPT: {
				for(auto spec : matches.getAllExcept()) {
					if(check(spec))
						goto no_match;
				}
				goto match;
			}
		}
				
		no_match:
			return nullptr;
		
		match:
			return impl -> getTarget(schemaId, methodId);
	}
	
	void buildStatus(kj::Vector<LoadBalancer::StatusInfo::Backend>& out) override {
		impl -> buildStatus(out);
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
	
	Maybe<capnp::Capability::Client> getTarget(uint64_t schemaId, uint16_t methodId) override {
		for(auto& rule : rules) {
			KJ_IF_MAYBE(pResult, rule.getTarget(schemaId, methodId)) {
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

struct LoadBalancerBase : public Ingress::Server {
	virtual RuleSet& getRules() = 0;
	
	capnp::Capability::Client selectCallTarget(uint64_t schemaId, uint16_t methodId) {		
		KJ_IF_MAYBE(pTarget, getRules().getTarget(schemaId, methodId)) {
			return *pTarget;
		}
		
		KJ_FAIL_REQUIRE("No suitable backend for method was available", methodId);
	}
	
	DispatchCallResult dispatchCall(
		uint64_t interfaceId, uint16_t methodId,
        capnp::CallContext<capnp::AnyPointer, capnp::AnyPointer> context
	) override {
		// Consume calls to Ingress interface
		// by redirecting to static dispatch
		if(interfaceId == capnp::typeId<Ingress>()) {
			return Ingress::Server::dispatchCall(interfaceId, methodId, mv(context));
		}
		
		auto params = context.getParams();
		capnp::MessageSize size = params.targetSize();
		
		auto tailRequest = selectCallTarget(interfaceId, methodId)
			.typelessRequest(interfaceId, methodId, size, capnp::Capability::Client::CallHints());
		tailRequest.set(params);
		
		return DispatchCallResult {
			context.tailCall(mv(tailRequest)),
			false, // isStreaming
			true   // allowCancellation
		};
	}
};

struct SubBalancer : public LoadBalancerBase {
	RuleSet& rules;
	
	SubBalancer(RuleSet& r) : rules(r) {}
	
	RuleSet& getRules() override { return rules; }
};
	
struct LoadBalancerImpl : public LoadBalancerBase, public kj::Refcounted, public LoadBalancer {	
	NetworkInterface::Client networkInterface;
	Temporary<LoadBalancerConfig> config;
	
	RuleSet ruleSet;
	
	kj::HashMap<kj::String, RuleSet> namedEndpoints;
	
	LoadBalancerImpl(NetworkInterface::Client networkInterface_, LoadBalancerConfig::Reader config_) :
		networkInterface(mv(networkInterface_)), config(config_),
		ruleSet(config, networkInterface)
	{
		for(auto ep : config.getNamedEndpoints()) {
			namedEndpoints.insert(kj::heapString(ep.getName().asReader()), RuleSet(config, networkInterface));
		}
	}
	
	RuleSet& getRules() override { return ruleSet; }
		
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
	
	Promise<void> getNamedEndpoint(GetNamedEndpointContext ctx) override {
		kj::StringPtr name = ctx.getParams().getName();
		KJ_IF_MAYBE(pRule, namedEndpoints.find(name)) {
			ctx.initResults().setEndpoint(kj::heap<SubBalancer>(*pRule).attach(addRef()));
			return READY_NOW;
		}
		
		return KJ_EXCEPTION(FAILED, "Named endpoint not found", name);
	}
};

}

Own<LoadBalancer> newLoadBalancer(NetworkInterface::Client clt, LoadBalancerConfig::Reader config) {
	return kj::refcounted<LoadBalancerImpl>(mv(clt), config);
}

}
