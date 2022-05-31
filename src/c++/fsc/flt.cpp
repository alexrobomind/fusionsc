#include "kernels-flt.h"
#include "cudata.h"

#include <fsc/flt.capnp.cu.h>

namespace fsc {
	void calc(Eigen::DefaultDevice& d) {
		capnp::MallocMessageBuilder mb;
		CupnpMessage<cu::FLTKernelData> kernelData(mb);
		CupnpMessage<cu::FLTKernelRequest> kernelRequest(mb);
		auto calculation = FSC_LAUNCH_KERNEL(
			fltKernel,
			
			d, 3,
			
			kernelData, kernelRequest
		);
	}
}