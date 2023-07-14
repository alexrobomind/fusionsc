#pragma once

#include "local.h"
#include "kernels.h"

#include <fsc/flt.capnp.h>

namespace fsc {	
	FLT::Client newFLT(Own<DeviceBase> device, FLTConfig::Reader config = FLTConfig::Reader());
}