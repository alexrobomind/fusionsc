#include "kernels-flt.h"
#include "cudata.h"

#include <fsc/flt.capnp.cu.h>

using namespace fsc;

namespace {
	
void validateField(ToroidalGrid::Reader grid, Float64Tensor::Reader data) {	
	auto shape = data.getShape();
	KJ_REQUIRE(shape.size() == 4);
	KJ_REQUIRE(shape[0] == grid.getNPhi());
	KJ_REQUIRE(shape[1] == grid.getNZ());
	KJ_REQUIRE(shape[2] == grid.getNR());
	KJ_REQUIRE(shape[3] == 3);
	
	KJ_REQUIRE(data.getData().size() == shape[0] * shape[1] * shape[2] * shape[3]);	
}
	
}

namespace fsc {
	void calc(Eigen::DefaultDevice& d) {
		capnp::MallocMessageBuilder mb;
		CupnpMessage<cu::FLTKernelData> kernelData(mb);
		CupnpMessage<cu::Float32Tensor> f32tensor(mb);
		CupnpMessage<cu::FLTKernelRequest> kernelRequest(mb);
		auto calculation = FSC_LAUNCH_KERNEL(
			fltKernel,
			
			d, 3,
			
			kernelData, f32tensor, kernelRequest
		);
	}
}