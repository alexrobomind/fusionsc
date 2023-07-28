#pragma once

#include "networking.h"

#include <capnp/capability.h>

#include <fsc/services.capnp.h>

namespace fsc {
	capnp::Capability::Client newLoadBalancer(NetworkInterface::Client, LoadBalancerConfig::Reader);
}
