#include "kernels-biotsavart.h"
#include "gpulaunch.h"

namespace fsc {
	namespace internal {

		INSTANTIATE_KERNEL(biotSavartKernel, ToroidalGridStruct, FilamentRef, double, double, double, FieldRef);

		template void addFields<Eigen::GpuDevice>(Eigen::GpuDevice&, FieldRef, FieldRef, double, Callback<>&&);

	}
}