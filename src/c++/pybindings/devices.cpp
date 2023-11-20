#include <fsc/devices/w7x.capnp.h>
#include <fsc/devices/w7x.h>

#include <fsc/devices/jtext.h>

#include "fscpy.h"


namespace fscpy {	
	void initDevices(py::module_& root) {
		py::module_ devices = root.def_submodule("devices", "Device-specific native functions");
		
		// ================ W7-X ==================
		
		py::module_ w7x = devices.def_submodule("w7x", "W7-X specific native functions");
		
		w7x.def("componentsDBResolver", &fsc::devices::w7x::newComponentsDBResolver);
		w7x.def("coilsDBResolver", &fsc::devices::w7x::newCoilsDBResolver);
		w7x.def("configDBResolver", &fsc::devices::w7x::newConfigDBResolver);
		
		w7x.def("geometryResolver", &fsc::devices::w7x::newW7xGeometryResolver);
		w7x.def("fieldResolver", &fsc::devices::w7x::newW7xFieldResolver);
		
		w7x.def("buildCoilFields",
			[](WithMessage<fsc::W7XCoilSet::Reader> coilSet, WithMessage<fsc::W7XCoilSet::Fields::Builder> out) {
				return fsc::devices::w7x::buildCoilFields(coilSet, out);
			}
		);
		
		// ================ J-TEXT ==================
		
		py::module_ jtext = devices.def_submodule("jtext", "J-TEXT specific native functions");
		
		jtext.def("geometryResolver", &fsc::devices::jtext::newGeometryResolver);
		jtext.def("fieldResolver", &fsc::devices::jtext::newFieldResolver);
		jtext.def("exampleGeqdsk", &fsc::devices::jtext::exampleGeqdsk);
	}
}