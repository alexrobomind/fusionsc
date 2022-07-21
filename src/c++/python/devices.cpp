#include <fsc/devices/w7x.capnp.h>
#include <fsc/devices/w7x.h>

#include "fscpy.h"

namespace {

fsc::ToroidalGrid::Reader w7xDefaultGrid() {
	return fsc::devices::w7x::DEFAULT_GRID;
}

}

namespace fscpy {
	void initDevices(py::module_& root) {
		py::module_ devices = root.def_submodule("devices");
		py::module_ w7x = devices.def_submodule("w7x");
		
		w7x.def("defaultGrid", &w7xDefaultGrid);
		w7x.def("offlineComponentsDB", &fsc::devices::w7x::newComponentsDBFromOfflineData);
		w7x.def("offlineCoilsDB", &fsc::devices::w7x::newCoilsDBFromOfflineData);
		
		w7x.def("webserviceCoilsDB", &fsc::devices::w7x::newCoilsDBFromWebservice);
		w7x.def("webserviceComponentsDB", &fsc::devices::w7x::newComponentsDBFromWebservice);
		
		w7x.def("componentsDBResolver", &fsc::devices::w7x::newComponentsDBResolver);
		w7x.def("coilsDBResolver", &fsc::devices::w7x::newCoilsDBResolver);
		
		w7x.def("buildCoilFields", &fsc::devices::w7x::buildCoilFields);
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