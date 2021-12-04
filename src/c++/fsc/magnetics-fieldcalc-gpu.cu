#include "magnetics-kernels.h"

namespace fsc { namespace internal {

INSTANTIATE_KERNEL(biotSavartKernel, ToroidalGridStruct, FilamentRef, double, double, double, FieldRef);

}}


#ifndef EIGEN_GPU_COMPILE_PHASE

#include "magnetics-inl.h"

namespace fsc { 

FieldCalculator::Client newGPUFieldCalculator(LibraryThread& lt) {
	using D = Eigen::GpuDevice;
	
	auto stream = kj::heap<Eigen::GpuStreamDevice>();
	auto dev  = kj::heap<Eigen::GpuDevice>(stream);
	dev = dev.attach(mv(stream));
	
	return FieldCalculator::Client(
		kj::heap<internal::FieldCalculatorImpl<D>>(lt, mv(dev))
	);
}

}

#endif