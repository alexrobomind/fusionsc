#include "fscpy.h"

#include <fsc/efit.h>
#include <fsc/offline.h>


namespace fscpy {
	
namespace {

	Promise<void> delay(double seconds) {
		uint64_t timeInNS = static_cast<uint64_t>(seconds * 1e9);
		auto targetPoint = kj::systemPreciseMonotonicClock().now() + timeInNS * kj::NANOSECONDS;
		
		return getActiveThread().timer().atTime(targetPoint);
	}
	
	Temporary<AxisymmetricEquilibrium> parseGFile(kj::StringPtr str) {
		Temporary<AxisymmetricEquilibrium> result;
		parseGeqdsk(result, str);
		return result;
	}

	void updateDataHelper(WithMessage<OfflineData::Builder> b, WithMessage<OfflineData::Reader> r) {
		updateOfflineData(b, r);
	}
	
	Temporary<VmecResults> loadVmecOutput(kj::StringPtr pathName) {
		Temporary<VmecResults> result;
		interpretOutputFile(kj::Path::evalNative(pathName), result.asBuilder());
		return result;
	}
}

void initHelpers(py::module_& m) {
	py::module_ timerModule = m.def_submodule("timer", "Timer helpers");
	timerModule.def("delay", &delay, "Creates a promise resolving at a defined delay (in seconds) after this function is called");
	
	py::module_ efitModule = m.def_submodule("efit", "EFIT processing helpers");
	efitModule.def("eqFromGFile", &parseGFile, "Creates an axisymmetric equilibrium from an EFIT GFile");
	
	py::module_ offlineModule = m.def_submodule("offline", "Offline data processing");
	
	py::module_ vmecModule = m.def_submodule("vmec");
	vmecModule.def("loadOutput", pathName);
	
	offlineModule.def("fieldResolver", &newOfflineFieldResolver);
	offlineModule.def("geometryResolver", &newOfflineGeometryResolver);
	offlineModule.def("updateOfflineData", &updateDataHelper);
}

}