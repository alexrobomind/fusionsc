#include "fscpy.h"

#include <fsc/store.h>
#include <fsc/magnetics.h>

#include <pybind11/pybind11.h>
#include <kj/common.h>
#include <capnp/dynamic.h>

using namespace fscpy;

kj::Own<py::dict> globalClasses;
kj::Own<py::type> baseType;
kj::Own<py::type> baseMetaType;

// ============================= Helper classes ===========================

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

void atExitFunction() {
	globalClasses = nullptr;
	baseType = nullptr;
	baseMetaType = nullptr;
}

auto pySimpleTokamak(double rMaj, double rMin, unsigned int nCoils, double Ip) {
	Temporary<MagneticField> result;
	simpleTokamak(result, rMaj, rMin, nCoils, Ip);
	return result;
}

void helperFunctions(py::module_& m) {
	m.def("simpleTokamak", &pySimpleTokamak, py::arg("rMaj") = 5.5, py::arg("rMin") = 1.5, py::arg("nCoils") = 25, py::arg("iP") = 0.2);
}

struct Simple {};

void bindHelperClasses(py::module_& m) {
	auto helpersModule = m.def_submodule("_helpers", "Internal helper classes");
	
	py::class_<BoundMethod>(helpersModule, "_BoundMethod")
		.def("__call__", &BoundMethod::call)
	;
	
	py::class_<UnknownObject>(helpersModule, "UnknownObject");
	
	py::class_<MethodDescriptor>(helpersModule, "MethodDescriptor", py::dynamic_attr())
		.def("__get__", &MethodDescriptor::get)
		.def("__repr__", [](py::object self) { return "Method"; })
	;
	
	py::class_<Simple>(helpersModule, "Simple", py::dynamic_attr())
		//.def("__str__", [](py::object self) { return py::hasattr(self, "desc") ? (py::str) self.attr("desc") : "<Unknown simple object>" ; })
		.def("__repr__", [](py::object self) { return py::hasattr(self, "desc") ? (py::str) self.attr("desc") : "<Unknown simple object>" ; })
	;
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
	// Creating a temporary ClientHook initializes a run-time
	// link between the capnp library and the capnp-rpc library
	(void) capnp::newBrokenCap("Don't look at me. I'm shy.");
	
	// Perform run-time initialization of python-related globals
	globalClasses = kj::heap<py::dict>();
	
	// Create global meta class
	baseType = kj::heap<py::type>(py::eval("type('FSCPyObject', (object,), {})"));
	py::type standardMeta    = py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject*>(&PyType_Type));
	/* py::type collectionsMeta = py::type::of(py::module_::import("collections.abc").attr("Mapping"));
	
	py::dict metaAttributes;
	metaAttributes["__module__"] = "fscpy";
	
	baseMetaType = kj::heap<py::type>(standardMeta(
		"MetaClass", py::make_tuple(collectionsMeta, standardMeta), metaAttributes
	));*/
	
	// Helper classes (we need these to define the standard metaclass)
	bindHelperClasses(m);
	helperFunctions(m);
	
	// baseMetaType = kj::heap<py::type>(standardMeta, py::type::of(py::type::of<Simple>()));
	baseMetaType  = kj::heap<py::type>(py::type::of(py::type::of<Simple>()));
	
	// Initialize bindings for all components
	initKj(m);
	initAsync(m);
	initCapnp(m);
	initData(m);
	initLoader(m);	
	initDevices(m);
	initService(m);
	
	// Load built-in schema
	loadDefaultSchema(m);
	
	auto atexitModule = py::module_::import("atexit");
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}