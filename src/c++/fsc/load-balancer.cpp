#include "load-balancer.h"
#include "data.h"

#include <random>

namespace fsc {

namespace {

struct LoadBalancerImpl : public capnp::Capability::Server {	
	struct Backend {
		LoadBalancerImpl& parent;
		kj::String url;
		
		Promise<void> maintenanceTask;
		
		bool connError = false;
		Maybe<capnp::Capability::Client> target = nullptr;
		kj::ListLink<Backend> link;
		
		Backend(LoadBalancerImpl& parent, kj::StringPtr url) :
			parent(parent),
			url(kj::heapString(url)),
			maintenanceTask(NEVER_DONE)
		{
			maintenanceNow();
		}
		
		~Backend() {
			if(target != nullptr) {
				parent.activeBackends.remove(*this);
			}
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
						return retryAfter(parent.config.getHeartbeatIntervalSeconds() * kj::SECONDS);
					},
					
					// Heartbeat call failed
					[this](kj::Exception error) {
						KJ_LOG(WARNING, "Lost connection to backend, reconnecting ...", url, error);
						parent.activeBackends.remove(*this);
						target = nullptr;
						return performMaintenance();
					}
				);
				
			} else {
				// Try to connect to target
				
				// Run connection
				auto connectRequest = parent.networkInterface.connectRequest();
				connectRequest.setUrl(url);
				
				return connectRequest.send()
				.then(
					[this](auto response) {
						connError = false;
						
						auto conn = response.getConnection();
						
						target = conn.getRemoteRequest().sendForPipeline().getRemote();
						parent.activeBackends.add(*this);
						
						KJ_LOG(INFO, "Connected to backend", url);
						
						return performMaintenance();
					},
					[this](kj::Exception error) {
						if(!connError) {
							KJ_LOG(WARNING, "Failed to connect to backend", url, error);
							connError = true;
						} else {
							KJ_LOG(INFO, "Failed to connect to backend", url, error);
						}

						return retryAfter(parent.config.getReconnectIntervalSeconds() * kj::SECONDS);
					}
				);
			}	
		}
		
		Promise<void> retryAfter(kj::Duration time) {
			return getActiveThread().timer().afterDelay(time)
			.then([this]() { return performMaintenance(); });
		}
		
		void maintenanceNow() {
			maintenanceTask = performMaintenance().eagerlyEvaluate([this](kj::Exception error) {
				KJ_LOG(ERROR, "Maintenance task for backend failed", url, error);
			});
		}
	};
	
	NetworkInterface::Client networkInterface;
	Temporary<LoadBalancerConfig> config;
	
	kj::Array<Backend> backends;
	kj::List<Backend, &Backend::link> activeBackends;
	
	std::mt19937 prng;
	
	uint32_t randomOffset = 0;
	uint32_t counter = 0;
	
	LoadBalancerImpl(NetworkInterface::Client networkInterface_, LoadBalancerConfig::Reader config_) :
		networkInterface(mv(networkInterface_)), config(config_),
		randomOffset(prng())
	{
		auto backendsIn = config.getBackends();
		
		auto backendsBuilder = kj::heapArrayBuilder<Backend>(backendsIn.size());
		for(auto node : backendsIn) {
			backendsBuilder.add(*this, node.getUrl());
		}
		
		backends = backendsBuilder.finish();
	}
	
	capnp::Capability::Client selectCallTarget() {
		if(activeBackends.size() == 0) {
			kj::throwFatalException(KJ_EXCEPTION(DISCONNECTED, "No active backend"));
		}
		
		// Adjust counter
		if(counter > activeBackends.size()) {
			counter = 0;
			randomOffset = prng();
		}
		
		uint32_t index = counter + randomOffset;
		++counter;
		
		index %= activeBackends.size();
		
		auto it = activeBackends.begin();
		for(auto i : kj::range(0, index))
			++it;
		
		auto maybeResult = it -> target;
		
		KJ_IF_MAYBE(pResult, maybeResult) {
			return *pResult;
		} else {
			KJ_LOG(ERROR, "Internal error, backend active but target not set");
			KJ_FAIL_REQUIRE("Load balancer failure due to invalid state");
		}
	}
	
	DispatchCallResult dispatchCall(
		uint64_t interfaceId, uint16_t methodId,
        capnp::CallContext<capnp::AnyPointer, capnp::AnyPointer> context
	) override {
		// TODO: Message size?
		auto tailRequest = selectCallTarget()
			.typelessRequest(interfaceId, methodId, nullptr, capnp::Capability::Client::CallHints());
		
		return DispatchCallResult {
			context.tailCall(mv(tailRequest)),
			false, // isStreaming
			true   // allowCancellation
		};
	}
	
};

}

capnp::Capability::Client newLoadBalancer(NetworkInterface::Client clt, LoadBalancerConfig::Reader config) {
	return kj::heap<LoadBalancerImpl>(mv(clt), config);
}

}