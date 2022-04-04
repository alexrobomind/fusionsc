#include "fscpy.h"

#include <pybind11/pybind11.h>
#include <kj/common.h>

using namespace fscpy;

kj::Own<py::dict> globalClasses;
kj::Own<py::type> baseType;
kj::Own<py::type> baseMetaType;

PYBIND11_MODULE(fscpy, m) {
	KJ_LOG(WARNING, "Initializing globals");
	globalClasses = kj::heap<py::dict>();
	baseType = kj::heap<py::type>(py::eval("type('FSCPyObject', (object,), {})"));
	baseMetaType = kj::heap<py::type>(py::eval("type"));
	
	KJ_LOG(WARNING, "Binding classes");
	bindKJClasses(m);
	bindCapnpClasses(m);
	
	KJ_LOG(WARNING, "Loading default schema");
	loadDefaultSchema(m);
}