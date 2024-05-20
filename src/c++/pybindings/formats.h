#pragma once

#include "fscpy.h"
#include "assign.h"
#include "capnp.h"

#include <fsc/textio.h>

namespace fscpy {
	namespace formats {
		using Language = textio::Dialect::Language;
		
		py::object dumps(py::object, Language, bool compact, bool asBytes);
		void dump(py::object, int, Language, bool compact);
		
		py::object read(py::object src, py::object dst);
	}
}
