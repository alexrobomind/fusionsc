#include "fscpy.h"

#include <fsc/data.h>

namespace fscpy {
	
void bindDataClasses(py::module_& m) {
	py::class_<LocalDataService>(m, "LocalDataService");
	py::class_<UnknownObject>(m, "UnknownObject");
}

}