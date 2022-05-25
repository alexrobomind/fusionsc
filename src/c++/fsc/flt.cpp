#include "kernels-flt.h"
#include "cudata.h"

#include <fsc/flt.capnp.cu.h>

namespace fsc {
	void calc(Eigen::DefaultDevice& d) {
		capnp::MallocMessageBuilder mb;
		CupnpMessage<cu::FLTKernelData> kernelData(mb);
		CupnpMessage<cu::FLTKernelRequest> kernelRequest(mb);
		TensorOpCost costEstimate(1e12, 1e12, 1e12);
		auto calculation = FSC_LAUNCH_KERNEL(
			fltKernel,
			
			_device, field.size(), costEstimate,
			
			kernelData, kernelRequest
		);
	}
}