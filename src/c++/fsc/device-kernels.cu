#include "kernels-biotsavart.h"
#include "kernels-flt.h"
#include "kernels.h"

namespace fsc {
	namespace internal {
		INSTANTIATE_KERNEL(kernels::biotSavartKernel, ToroidalGridStruct, kernels::FilamentRef, double, double, double, kernels::FieldRef);
		INSTANTIATE_KERNEL(kernels::addFieldKernel, kernels::FieldRef, kernels::FieldRef, double);
		INSTANTIATE_KERNEL(fsc::fltKernel, fsc::cu::FLTKernelData, TensorMap<Tensor<double, 4>>, const fsc::cu::FLTKernelRequest);
	}
}