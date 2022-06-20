#include "kernels.h"
#include "flt-kernels.h"

namespace fsc { namespace internal {
	INSTANTIATE_KERNEL(
		fsc::fltKernel,
	
		CuPtr<fsc::cu::FLTKernelData>,
		Eigen::TensorMap<Eigen::Tensor<double, 4>>,
		CuPtr<fsc::cu::FLTKernelRequest>
	);
}}