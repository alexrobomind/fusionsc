#include "kernels.h"
#include "flt-kernels.h"

namespace fsc { namespace internal {
	INSTANTIATE_KERNEL(
		fsc::fltKernel,
	
		fsc::CuPtr<fsc::cu::FLTKernelData>,
		Eigen::TensorMap<Eigen::Tensor<double, 4>>,
		fsc::CuPtr<fsc::cu::FLTKernelRequest>,
		
		fsc::CuPtr<const fsc::cu::MergedGeometry>,
		fsc::CuPtr<const fsc::cu::IndexedGeometry>,
		fsc::CuPtr<const fsc::cu::IndexedGeometry::IndexData>
	);
}}