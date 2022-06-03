#include "fscpy.h"

#include <pybind11/pybind11.h>
#include <kj/common.h>

using namespace fscpy;

kj::Own<py::dict> globalClasses;
kj::Own<py::type> baseType;
kj::Own<py::type> baseMetaType;

namespace {

struct BoundMethod {
	py::object method;
	py::object self;
	
	BoundMethod(py::object method, py::object self) : method(method), self(self) {};
	
	py::object call(py::args args, py::kwargs kwargs) { return method(self, *args, **kwargs); }
};

struct MethodDescriptor {
	py::object target;
	
	MethodDescriptor(py::object target) : target(target) {}
	py::object get(py::object self, py::object objtype) { if(self.is_none()) return target; return py::cast(BoundMethod(target, self)); }
};

struct Simple {		
};

void atExitFunction() {
	globalClasses = nullptr;
	baseType = nullptr;
	baseMetaType = nullptr;
}

}

namespace fscpy {

py::object methodDescriptor(py::object method) {
	return py::cast(MethodDescriptor(method));
}

py::object simpleObject() {
	return py::cast(Simple());
}

}

PYBIND11_MODULE(fscpy, m) {
	py::class_<BoundMethod>(m, "_BoundMethod")
		.def("__call__", &BoundMethod::call)
	;
	
	py::class_<MethodDescriptor>(m, "_MethodDescriptor", py::dynamic_attr())
		.def("__get__", &MethodDescriptor::get)
		.def("__repr__", [](py::object self) { return "Method"; })
	;
	
	py::class_<Simple>(m, "_Simple", py::dynamic_attr())
		//.def("__str__", [](py::object self) { return py::hasattr(self, "desc") ? (py::str) self.attr("desc") : "<Unknown simple object>" ; })
		.def("__repr__", [](py::object self) { return py::hasattr(self, "desc") ? (py::str) self.attr("desc") : "<Unknown simple object>" ; })
	;
	
	globalClasses = kj::heap<py::dict>();
	baseType = kj::heap<py::type>(py::eval("type('FSCPyObject', (object,), {})"));
	baseMetaType = kj::heap<py::type>(py::eval("type"));
	
	bindKJClasses(m);
	bindCapnpClasses(m);
	bindAsyncClasses(m);
	bindDataClasses(m);
	
	loadDefaultSchema(m);
	
	auto atexitModule = py::module_::import("atexit");
	
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}