#include <fsc/devices/w7x.capnp.h>
#include <fsc/devices/w7x.h>

#include <fsc/devices/jtext.h>

#include "fscpy.h"


namespace fscpy {
	namespace {
		WithMessage<fsc::ToroidalGrid::Reader> w7xDefaultGrid() {
			return noMessage(fsc::devices::w7x::DEFAULT_GRID.get());
		}

		WithMessage<fsc::CartesianGrid::Reader> w7xDefaultGeometryGrid() {
			return noMessage(fsc::devices::w7x::DEFAULT_GEO_GRID.get());
		}
	}
	
	void initDevices(py::module_& root) {
		py::module_ devices = root.def_submodule("devices", "Device-specific native functions");
		
		// ================ W7-X ==================
		
		py::module_ w7x = devices.def_submodule("w7x", "W7-X specific native functions");
		
		w7x.def("defaultGrid", &w7xDefaultGrid);
		w7x.def("defaultGeometryGrid", &w7xDefaultGeometryGrid);		
		
		w7x.def("webserviceCoilsDB", &fsc::devices::w7x::newCoilsDBFromWebservice);
		w7x.def("webserviceComponentsDB", &fsc::devices::w7x::newComponentsDBFromWebservice);
		
		w7x.def("componentsDBResolver", &fsc::devices::w7x::newComponentsDBResolver);
		w7x.def("coilsDBResolver", &fsc::devices::w7x::newCoilsDBResolver);
		w7x.def("configDBResolver", &fsc::devices::w7x::newConfigDBResolver);
		
		w7x.def("geometryResolver", &fsc::devices::w7x::newW7xGeometryResolver);
		w7x.def("fieldResolver", &fsc::devices::w7x::newW7xFieldResolver);
		
		w7x.def("buildCoilFields", &fsc::devices::w7x::buildCoilFields);
		
		// ================ J-TEXT ==================
		
		py::module_ jtext = devices.def_submodule("jtext", "J-TEXT specific native functions");
		
		jtext.def("geometryResolver", &fsc::devices::jtext::newGeometryResolver);
		jtext.def("fieldResolver", &fsc::devices::jtext::newFieldResolver);
		jtext.def("exampleGeqdsk", &fsc::devices::jtext::exampleGeqdsk);
		
		jtext.def("defaultGrid", &fsc::devices::jtext::defaultGrid);
		jtext.def("defaultGeometryGrid", &fsc::devices::jtext::defaultGeometryGrid);
	}
	
	void loadDeviceSchema(py::module_& m) {
		py::module_ devices = m.attr("devices");
		py::module_ w7x = devices.attr("w7x");
		
		defaultLoader.addBuiltin<
			devices::w7x::CoilsDB,
			devices::w7x::ComponentsDB
		>();
		
		auto schemas = getBuiltinSchemas<devices::w7x::CoilsDB, devices::w7x::ComponentsDB>();
			
		for(auto node : schemas) {
			defaultLoader.importNodeIfRoot(node.getId(), w7x);
		}
	}
}