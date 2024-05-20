#include "fscpy.h"

#include <fsc/store.h>
#include <fsc/magnetics.h>

#include <pybind11/pybind11.h>
#include <kj/common.h>
#include <kj/exception.h>
#include <capnp/dynamic.h>

#include <cstdlib>

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
	
	py::class_<ContiguousCArray>(helpersModule, "ContiguousCArray", py::buffer_protocol())
		.def_buffer(&ContiguousCArray::getBuffer)
	;
}

}

namespace fscpy {

py::buffer_info ContiguousCArray::getBuffer() {
	KJ_REQUIRE(format.size() > 0, "Format string must be specified before requesting buffer");
	
	// Compute C strides
	std::vector<py::ssize_t> strides(shape.size());
	size_t stride = elementSize;
	for(int i = shape.size() - 1; i >= 0; --i) {
		strides[i] = stride;
		stride *= shape[i];
	}
	
	return py::buffer_info(
		data.begin(), elementSize, format.cStr(), shape.size(), shape, strides, /* readonly = */ false
	);
}

py::object methodDescriptor(py::object method) {
	return py::cast(MethodDescriptor(method));
}

py::object simpleObject() {
	return py::cast(Simple());
}

}

PYBIND11_MODULE(native, m) {
	// Creating a temporary ClientHook initializes a run-time
	// link between the capnp library and the capnp-rpc library
	(void) capnp::newBrokenCap("Don't look at me. I'm shy.");
	
	/* Note: The following call makes sense in standalone
	   programs, but pybind11 and python have their own
	   crash handling infrastructure, and setting this up
	   conflicts with python's Ctrl+C handler.
	   */
	// kj::printStackTraceOnCrash();
	
	// There exists no public interface for the following code yet :(
	const char* envVal = std::getenv("FUSIONSC_VERBOSE");
	if(envVal != nullptr && std::strcmp(envVal, "0") != 0) {
		kj::_::Debug::setLogLevel(kj::LogSeverity::INFO);
	}
	
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
	
	// Retrieve the pybind11 metaclass
	baseMetaType  = kj::heap<py::type>(py::type::of(py::type::of<Simple>()));
	
	// Initialize bindings for all components
	initKj(m);
	initAsync(m);
	initCapnp(m);
	initData(m);
	initLoader(m);	
	initDevices(m);
	initService(m);
	initHelpers(m);
	initStructio(m);
	
	auto atexitModule = py::module_::import("atexit");
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}