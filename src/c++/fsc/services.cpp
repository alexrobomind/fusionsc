#include "services.h"
#include "magnetics.h"
#include "kernels.h"

using namespace fsc;

namespace {
	
template<typename T>
auto selectDevice(T t, WorkerType preferredType) {
	#ifdef FSC_WITH_CUDA
	
	try {
		if(preferredType == WorkerType::GPU) {
			return tuple(t(newGpuDevice()), WorkerType::GPU);
		}
	} catch(kj::Exception& e) {
	}
	
	#endif
	
	return tuple(t(newThreadPoolDevice()), WorkerType::CPU);
}

struct RootServer : public RootService::Server {
	RootServer(LibraryThread& lt, RootConfig::Reader config) :
		lt(lt->addRef())
	{}
	
	Promise<void> newFieldCalculator(NewFieldCalculatorContext context) {
		auto factory = [this, context](auto device) mutable {			
			return ::fsc::newFieldCalculator(lt, context.getParams().getGrid(), mv(device));
		};
		
		auto selectResult = selectDevice(factory, context.getParams().getPreferredDeviceType());
		
		auto results = context.getResults();
		results.setCalculator(kj::get<0>(selectResult));
		results.setDeviceType(kj::get<1>(selectResult));
		
		return READY_NOW;
	}
	
private:
	LibraryThread lt;
};

}

RootService::Client fsc::createRoot(LibraryThread& lt, RootConfig::Reader config) {
	return kj::heap<RootServer>(lt, config);
}