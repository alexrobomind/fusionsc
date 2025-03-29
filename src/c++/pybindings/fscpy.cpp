#include "fscpy.h"

#include <fsc/store.h>

#include <pybind11/pybind11.h>
#include <kj/common.h>
#include <kj/exception.h>
#include <capnp/dynamic.h>

#include <cstdlib>

#ifdef WIN32
#include <crtdbg.h>
#endif

using namespace fscpy;

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
	
	py::class_<ContiguousCArray>(helpersModule, "ContiguousCArray", py::buffer_protocol())
		.def_buffer(&ContiguousCArray::getBuffer)
	;
}

}

namespace fscpy {

ContiguousCArray::~ContiguousCArray() {
	if(data.size() == 0) return;
	if(format == "O") {
		auto ptrView = this -> template as<PyObject*>();
		
		for(auto e : ptrView) {
			Py_XDECREF(e);
		}
	}
}

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

}

namespace {

#ifdef WIN32

int crtReportHook(int nRptType, char *szMsg, int *retVal) {
	if(nRptType == _CRT_ASSERT) {
		KJ_LOG(WARNING, "Win32 assertion failed", kj::getStackTrace());
		// KJ_FAIL_REQUIRE("Assertion failed", szMsg);
	}
	return 0;
}

#endif

}

PYBIND11_MODULE(native, m) {
	// Creating a temporary ClientHook initializes a run-time
	// link between the capnp library and the capnp-rpc library
	(void) capnp::newBrokenCap("Don't look at me. I'm shy.");
	
	#ifdef WIN32
	#ifdef _DEBUG
		_CrtSetReportHook(&crtReportHook);
	#endif
	#endif
	
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
	
	// Helper classes (we need these to define the standard metaclass)
	bindHelperClasses(m);	
	
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
}
