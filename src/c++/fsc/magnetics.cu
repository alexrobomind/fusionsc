#include "kernels.h"
#include "magnetics-kernels.h"

namespace fsc { namespace internal {
	INSTANTIATE_KERNEL(
		kernels::biotSavartKernel,
		
		ToroidalGridStruct,
		kernels::FilamentRef,
		double,
		double,
		double,
		kernels::FieldRef
	);
	
	INSTANTIATE_KERNEL(
		kernels::addFieldKernel,
		kernels::FieldRef,
		kernels::FieldRef,
		double
	);
	
	INSTANTIATE_KERNEL(
		kernels::addFieldInterpKernel,
		const unsigned int,
		kernels::FieldRef,
		ToroidalGridStruct,
		kernels::FieldRef,
		ToroidalGridStruct,
		double
	);
}}