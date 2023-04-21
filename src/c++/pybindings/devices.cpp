#include <fsc/devices/w7x.capnp.h>
#include <fsc/devices/w7x-geometry.capnp.h>
#include <fsc/devices/w7x.h>

#include <fsc/devices/jtext.h>

#include "fscpy.h"

namespace {

fsc::ToroidalGrid::Reader w7xDefaultGrid() {
	return fsc::devices::w7x::DEFAULT_GRID;
}

fsc::CartesianGrid::Reader w7xDefaultGeometryGrid() {
	return fsc::devices::w7x::DEFAULT_GEO_GRID;
}

fsc::Geometry::Reader w7xOp21BafflesNoHoles() {
	return fsc::devices::w7x::W7X_OP21_BAFFLES_NO_HOLES;
}

fsc::Geometry::Reader w7xOp21HeatShieldNoHoles() {
	return fsc::devices::w7x::W7X_OP21_HEAT_SHIELD_NO_HOLES;
}

fsc::Geometry::Reader w7xOp21Divertor() {
	return fsc::devices::w7x::W7X_OP21_DIVERTOR;
}

}

namespace fscpy {
	void initDevices(py::module_& root) {
		py::module_ devices = root.def_submodule("devices");
		py::module_ w7x = devices.def_submodule("w7x");
		
		w7x.def("defaultGrid", &w7xDefaultGrid);
		w7x.def("defaultGeometryGrid", &w7xDefaultGeometryGrid);
		
		w7x.def("op21BafflesNoHoles", &w7xOp21BafflesNoHoles);
		w7x.def("op21HeatShieldNoHoles", &w7xOp21HeatShieldNoHoles);
		w7x.def("op21Divertor", &w7xOp21Divertor);		
		
		w7x.def("webserviceCoilsDB", &fsc::devices::w7x::newCoilsDBFromWebservice);
		w7x.def("webserviceComponentsDB", &fsc::devices::w7x::newComponentsDBFromWebservice);
		
		w7x.def("componentsDBResolver", &fsc::devices::w7x::newComponentsDBResolver);
		w7x.def("coilsDBResolver", &fsc::devices::w7x::newCoilsDBResolver);
		w7x.def("configDBResolver", &fsc::devices::w7x::newCoilsDBResolver);
		
		w7x.def("fieldResolver", &fsc::devices::w7x::newW7xResolver);
		
		w7x.def("buildCoilFields", &fsc::devices::w7x::buildCoilFields);
		
		py::module_ jtext = devices.def_submodule("jtext");
		
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