#include "kernels-biotsavart.h"
#include "kernels.h"

namespace fsc {
	namespace internal {
		INSTANTIATE_KERNEL(kernels::biotSavartKernel, ToroidalGridStruct, FilamentRef, double, double, double, FieldRef);
		INSTANTIATE_KERNEL(kernels::addFieldKernel, FieldRef, FieldRef, double);
	}
}