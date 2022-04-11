#pragma once

#include <pybind11/pybind11.h>
#include <fsc/common.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, kj::Own<T>);

namespace py = pybind11;

namespace fscpy {
	using namespace fsc;
	
	void bindCapnpClasses(py::module_& m);
	void bindKJClasses(py::module_& m);
	void loadDefaultSchema(py::module_& m);
	
	constexpr inline kj::StringPtr INTERNAL_ACCESS_KEY = "9821736419873251235"_kj;
	
	py::object methodDescriptor(py::object method);
}

extern kj::Own<py::dict> globalClasses;
extern kj::Own<py::type> baseType;
extern kj::Own<py::type> baseMetaType;

// Custom type casters
#include "typecast_kj.h"
#include "typecast_capnp.h"