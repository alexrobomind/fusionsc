#include "services.h"
#include "magnetics.h"

using namespace fsc;

namespace {

struct RootServer : public RootService::Server {
	RootServer(LibraryThread& lt, RootConfig::Reader config) :
		lt(lt->addRef())
	{}
	
	Promise<void> newFieldCalculator(NewFieldCalculatorContext context) {
		context.getResults().setCalculator(newCPUFieldCalculator(lt));
		return READY_NOW;
	}
	
private:
	LibraryThread lt;
};

}

RootService::Client fsc::createRoot(LibraryThread& lt, RootConfig::Reader config) {
	return kj::heap<RootServer>(lt, config);
}