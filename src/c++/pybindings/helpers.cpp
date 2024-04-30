#include "fscpy.h"
#include "async.h"
#include "serialize.h"

#include <fsc/efit.h>
#include <fsc/offline.h>
#include <fsc/vmec.h>
#include <fsc/geometry.h>

#include <fsc/data.h>

#include <fsc/vmec.capnp.h>



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
	
	Temporary<VmecResult> loadVmecOutput(kj::StringPtr pathName) {
		Temporary<VmecResult> result;
		interpretOutputFile(getActiveThread().filesystem().getCurrentPath().evalNative(pathName), result.asBuilder());
		return result;
	}
	
	FieldResolver::Client fieldResolver(DynamicCapabilityClient ref) {
		return newOfflineFieldResolver(ref.castAs<DataRef<OfflineData>>());
	}
	
	GeometryResolver::Client geometryResolver(DynamicCapabilityClient ref) {
		return newOfflineGeometryResolver(ref.castAs<DataRef<OfflineData>>());
	}
	
	Promise<void> writePlyHelper(WithMessage<Geometry::Reader> geo, kj::StringPtr filename, bool binary) {
		return writePly(geo, filename, binary);
	}
}

void initHelpers(py::module_& m) {
	py::module_ timerModule = m.def_submodule("timer", "Timer helpers");
	timerModule.def("delay", &delay, "Creates a promise resolving at a defined delay (in seconds) after this function is called");
	
	py::module_ efitModule = m.def_submodule("efit", "EFIT processing helpers");
	efitModule.def("eqFromGFile", &parseGFile, "Creates an axisymmetric equilibrium from an EFIT GFile");
	
	py::module_ offlineModule = m.def_submodule("offline", "Offline data processing");
	offlineModule.def("fieldResolver", &fieldResolver);
	offlineModule.def("geometryResolver", &geometryResolver);
	offlineModule.def("updateOfflineData", &updateDataHelper);
	
	py::module_ vmecModule = m.def_submodule("vmec");
	vmecModule.def("loadOutput", &loadVmecOutput);
	
	py::module_ geometryModule = m.def_submodule("geometry");
	geometryModule.def(
		"writePly", &writePlyHelper,
		py::arg("geometry"), py::arg("filename"), py::arg("binary") = true
	);
	geometryModule.def(
		"readPly", &readPly, py::arg("filename"), py::arg("maxVertices") = 0, py::arg("maxIndices") = 0
	);
	
	py::module_ serializeModule = m.def_submodule("serialize");
	serializeModule.def("loadEnumArray", &loadEnumArray);
	serializeModule.def("loadStructArray", &loadStructArray);
}

}
