#pragma once

#include "fscpy.h"
#include "assign.h"
#include "capnp.h"

#include <fsc/structio.h>

namespace fscpy {
	namespace structio {		
		using Language = fsc::structio::Dialect::Language;
		
		py::object dumps(py::object, Language, bool compact, bool asBytes);
		void dump(py::object, int, Language, bool compact);
		
		py::object read(py::object src, py::object dst);
	}
}
