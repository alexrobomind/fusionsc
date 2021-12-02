#include "magnetics.h"
#include "tensor.h"
#include "data.h"
#include "magnetics-inl.h"

#include <cmath>

#include <kj/map.h>
#include <kj/refcount.h>


namespace fsc { 
template<typename Device>
struct FieldCalculatorImpl : public FieldCalculator::Server {
	LibraryThread lt;
	
	FieldCalculatorImpl(LibraryThread& lt) :
		lt(lt -> addRef())
	{}
	
	Promise<void> get(GetContext context) {
		auto device = kj::heap<Eigen::DefaultDevice>();
		
		FieldCalculationSession::Client newClient(
			kj::heap<internal::CalculationSession<Device>>(
				*device,
				context.getParams().getGrid(),
				lt
			).attach(mv(device))
		);
		context.getResults().setSession(newClient);
		return READY_NOW;
	}
};

FieldCalculator::Client newCPUFieldCalculator(LibraryThread& lt) {
	return FieldCalculator::Client(
		kj::heap<FieldCalculatorImpl<Eigen::DefaultDevice>>(lt)
	);
}

}