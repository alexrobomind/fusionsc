#include "magnetics.h"
#include "tensor.h"
#include "data.h"
#include "magnetics-inl-kernel.h"

#include <cmath>

#include <kj/map.h>
#include <kj/refcount.h>


namespace fsc { 

FieldCalculator::Client newGPUFieldCalculator(LibraryThread& lt) {
	using D = Eigen::GpuDevice;
	
	auto stream = kj::heap<Eigen::GpuStreamDevice>();
	auto dev    = kj::heap<Eigen::GpuDevice>(stream);
	
	dev = dev.attach(mv(stream));
	
	return FieldCalculator::Client(
		kj::heap<internal::FieldCalculatorImpl<D>>(lt, mv(dev))
	);
}

}