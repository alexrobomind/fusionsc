#include "fscpy.h"

#include <pybind11/pybind11.h>
#include <kj/common.h>

using namespace fscpy;

kj::Own<py::dict> globalClasses;
kj::Own<py::type> baseType;
kj::Own<py::type> baseMetaType;

struct BoundMethod {
	py::object method;
	py::object self;
	
	BoundMethod(py::object method, py::object self) : method(method), self(self) {};
	
	py::object call(py::args args, py::kwargs kwargs) { return method(self, *args, **kwargs); }
};

struct MethodDescriptor {
	py::object target;
	
	MethodDescriptor(py::object target) : target(target) {}
	BoundMethod get(py::object self, py::object objtype) { return BoundMethod(target, self); }
};

namespace fscpy {

py::object methodDescriptor(py::object method) {
	return py::cast(MethodDescriptor(method));
}

}

PYBIND11_MODULE(fscpy, m) {
	py::class_<BoundMethod>(m, "BoundMethod")
		.def("__call__", &BoundMethod::call)
	;
	
	py::class_<MethodDescriptor>(m, "MethodDescriptor")
		.def("__get__", &MethodDescriptor::get)
	;
	
	globalClasses = kj::heap<py::dict>();
	baseType = kj::heap<py::type>(py::eval("type('FSCPyObject', (object,), {})"));
	baseMetaType = kj::heap<py::type>(py::eval("type"));
	
	bindKJClasses(m);
	bindCapnpClasses(m);
	bindAsyncClasses(m);
	
	loadDefaultSchema(m);
}