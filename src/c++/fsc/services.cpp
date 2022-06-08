#include "services.h"
#include "magnetics.h"
#include "kernels.h"

using namespace fsc;

namespace {
	
template<typename T>
auto selectDevice(T t, DeviceType preferredType) {
	#ifdef FSC_WITH_CUDA
	
	try {
		if(preferredType == DeviceType::Gpu) {
			return make_tuple(t(newGpuDevice()), DeviceType::Gpu);
		}
	} catch(kj::Exception& e) {
	}
	
	#endif
	
	return make_tuple(t(newCpuDevice()), DeviceType::Cpu);
}

struct RootServer : public RootService::Server {
	RootServer(LibraryThread& lt, RootConfig::Reader config) :
		lt(lt->addRef())
	{}
	
	Promise<void> newFieldCalculator(NewFieldCalculatorContext context) {
		auto factory = [context](auto device) {
			using EigenDevice = decltype(*device);
			
			return newFieldCalculator(lt, context.getParams().getGrid(), device);
		};
		
		FieldCalculator::Client result;
		DeviceType type;
		
		refTuple(result, type) = selectDevice(factory, context.getParams().getPreferredDeviceType());
		
		auto results = context.getResults();
		results.setCalculator(result);
		results.setDeviceType(type);
		
		return READY_NOW;
	}
	
private:
	LibraryThread lt;
};

}

RootService::Client fsc::createRoot(LibraryThread& lt, RootConfig::Reader config) {
	return kj::heap<RootServer>(lt, config);
}