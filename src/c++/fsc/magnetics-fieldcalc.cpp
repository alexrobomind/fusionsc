#include "magnetics.h"
#include "tensor.h"
#include "data.h"
#include "magnetics-inl.h"

#include <cmath>

#include <kj/map.h>
#include <kj/refcount.h>


namespace fsc { 

FieldCalculator::Client newCPUFieldCalculator(LibraryThread& lt) {
	using D = Eigen::ThreadPoolDevice;
	
	auto pool = kj::heap<Eigen::ThreadPool>(numThreads());
	auto dev  = kj::heap<Eigen::ThreadPoolDevice>(pool.get(), numThreads());
	dev = dev.attach(mv(pool));
	
	return FieldCalculator::Client(
		kj::heap<internal::FieldCalculatorImpl<D>>(lt, mv(dev))
	);
	/*return FieldCalculator::Client(
		kj::heap<FieldCalculatorImpl<Eigen::DefaultDevice>>(lt, kj::heap<Eigen::DefaultDevice>())
	);*/
}

}