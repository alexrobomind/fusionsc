#include "fscpy.h"
#include "async.h"
#include "serialize.h"

#include <fsc/efit.h>
#include <fsc/offline.h>
#include <fsc/vmec.h>
#include <fsc/geometry.h>

#include <fsc/data.h>

#include <fsc/vmec.capnp.h>

#include <fsc/capi-lvn.h>
#include <fsc/services.h>

#include <pybind11/numpy.h>



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
	
	void writePlyHelper(WithMessage<MergedGeometry::Reader> geo, kj::StringPtr filename, bool binary) {
		return writePly(geo, filename, binary);
	}
	
	Promise<py::object> exportRawHelper(WithMessage<Geometry::Reader> geo, bool triangulate) {
		KJ_REQUIRE(geo.isMerged(), "Can only export merged geometries as raw");
		
		return getActiveThread().dataService().download(geo.getMerged())
		.then([triangulate](auto mergedGeo) -> py::object {
			auto exported = exportRaw(mergedGeo.get(), triangulate);
						
			auto& points = kj::get<0>(exported);
			auto& polys = kj::get<1>(exported);
			
			double* rawData = (double*) points.begin();
			
			std::array<size_t, 2> pointsShape = {points.size(), 3};
			py::array_t<double> pointsOut(pointsShape, rawData, unknownObject(mv(points)));
			
			py::list polysOut;
			for(auto& polyIn : polys) {
				py::list polyOut(polyIn.size());
				for(auto i : kj::indices(polyIn)) {
					polyOut[i] = polyIn[i];
				}
				polysOut.append(mv(polyOut));
			}
			
			return py::make_tuple(mv(pointsOut), mv(polysOut));
		});
	}
	
	Temporary<Geometry> importRawHelper(py::object points, py::object indices) {
		auto pointsArray = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(points);
		
		KJ_REQUIRE(pointsArray.ndim() == 2);
		KJ_REQUIRE(pointsArray.shape(1) == 3);
		
		kj::ArrayPtr<std::array<const double, 3>> pointsData(
			(std::array<const double, 3>*) pointsArray.data(0, 0),
			pointsArray.shape(0)
		);
		
		auto idxBuilder = kj::heapArrayBuilder<kj::Array<const size_t>>(py::len(indices));
		for(auto poly : indices) {
			auto faceBuilder = kj::heapArrayBuilder<const size_t>(py::len(poly));
			
			for(auto idx : poly) {
				faceBuilder.add(py::cast<size_t>(idx));
			}
			
			idxBuilder.add(faceBuilder.finish());
		}
		
		Temporary<MergedGeometry> merged;
		importRaw(pointsData, idxBuilder.finish(), merged);
		
		Temporary<Geometry> result;
		result.setMerged(getActiveThread().dataService().publish(merged.asReader()));
		
		return result;
	}
	
	Promise<void> writeMgridHelper(ComputedField::Reader cf, kj::StringPtr fileName) {
		auto& fs = getActiveThread().filesystem();
		
		auto fsPath = fs.getCurrentPath().evalNative(fileName);
		return writeMGridFile(fsPath, cf);
	}
	
	py::capsule getStoreHelper() {
		auto storeDestructor = [](PyObject* capsulePtr) {
			void* rawPtr = PyCapsule_GetPointer(capsulePtr, "fusionsc_DataStore");
			auto ptr = static_cast<fusionsc_DataStore*>(rawPtr);
			ptr -> decRef(ptr);
		};
		
		auto& store = PythonContext::libraryThread() -> store();
		
		return py::capsule(store.incRef(), "fusionsc_DataStore", storeDestructor);
	}
	
	struct LocalServer {
		py::capsule backend;
		capnp::InterfaceSchema schema;
		
		LocalServer(py::capsule c, capnp::InterfaceSchema is) : backend(mv(c)), schema(mv(is)) {}
		
		capnp::DynamicCapability::Client connect() {
			KJ_REQUIRE(kj::StringPtr(backend.name()) == "fusionsc_LvnHub*");
			fusionsc_LvnHub** ppHub = backend;
			
			return connectInProcess(*ppHub).castAs<capnp::DynamicCapability>(schema);
		}
	};		
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
	vmecModule.def("writeMGrid", &writeMgridHelper);
	
	py::module_ geometryModule = m.def_submodule("geometry");
	geometryModule.def(
		"writePly", &writePlyHelper,
		py::arg("geometry"), py::arg("filename"), py::arg("binary") = true
	);
	geometryModule.def(
		"readPly", &readPly, py::arg("filename")
	);
	geometryModule.def(
		"exportRaw", &exportRawHelper,
		py::arg("geometry"), py::arg("triangulate") = true
	);
	geometryModule.def(
		"importRaw", &importRawHelper,
		py::arg("points"), py::arg("polygons")
	);
	
	py::module_ serializeModule = m.def_submodule("serialize");
	serializeModule.def("loadEnumArray", &loadEnumArray);
	serializeModule.def("loadStructArray", &loadStructArray);
	
	py::module_ localModule = m.def_submodule("local");
	localModule.def("getStore", &getStoreHelper);
	
	py::class_<LocalServer>(localModule, "LocalServer")
		.def(py::init<py::capsule, capnp::InterfaceSchema>(), py::attr("lvnCapsule"), py::attr("schema"))
		.def("connect", &LocalServer::connect)
	;
}

}
