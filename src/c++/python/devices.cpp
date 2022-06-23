#include <fsc/devices/w7x.capnp.h>

#include "fscpy.h"

namespace {

fsc::ToroidalGrid::Reader w7xDefaultGrid() {
	return fsc::devices::w7x::DEFAULT_GRID;
}

}

namespace fscpy {
	void bindDevices(py::module_& root) {
		py::module_ devices = root.def_submodule("devices");
		py::module_ w7x = devices.def_submodule("w7x");
		
		w7x.def("defaultGrid", &w7xDefaultGrid);
	}
}