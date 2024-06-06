#pragma once

#include "local.h"
#include "kernels/device.h"

#include <fsc/flt.capnp.h>

namespace fsc {	
	Own<FLT::Server> newFLT(Own<DeviceBase> device, FLTConfig::Reader config = FLTConfig::Reader());
}
