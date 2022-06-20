#pragma once

#include "local.h"
#include "kernels.h"

#include <fsc/flt.capnp.h>

namespace fsc {
	
	// TODO: Make this accept data service instead	
	FLT::Client newFLT(LibraryThread& lt, Own<Eigen::ThreadPoolDevice> device);
	
	#ifdef FSC_WITH_CUDA
	
	FLT::Client newFLT(LibraryThread& lt, Own<Eigen::GpuDevice> device);
	
	#endif
}