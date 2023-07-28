#pragma once

#include "networking.h"

#include <capnp/capability.h>

#include <fsc/services.capnp.h>

namespace fsc {
	struct LoadBalancer {
		struct StatusInfo {
			struct Backend {
				enum Status { OK, DISCONNECTED };
				
				kj::String url;
				Status status;
			};
				
			kj::Array<Backend> backends;
		};
		
		virtual Own<LoadBalancer> addRef() = 0;
		virtual StatusInfo status() = 0;
		virtual capnp::Capability::Client loadBalanced() = 0;
		
		inline virtual ~LoadBalancer() noexcept(false) {}
	};
	
	Own<LoadBalancer> newLoadBalancer(NetworkInterface::Client, LoadBalancerConfig::Reader);
}
