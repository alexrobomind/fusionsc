#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <fsc/common.h>

// Custom type casters
#include "typecast_kj.h"
#include "typecast_capnp.h"

namespace py = pybind11;

namespace fscpy {
	using namespace fsc;
	
	void defCapnpClasses(py::module_ m);
	void dynamicValueBindings(py::module_& m);	
}

extern kj::Own<py::dict> globalClasses;
