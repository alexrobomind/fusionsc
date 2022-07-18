#include "async.h"

#include <capnp/dynamic.h>

#include <fsc/data.h>

namespace {
	fsc::LocalDataService& dataService()  {
		return fscpy::PyContext::libraryThread()->dataService();
	}

	void atExitFunction() {
		fscpy::PyContext::library()->stopSteward();
	}
	
	void cycle() {
		fscpy::PyContext::libraryThread()->waitScope().poll();
	}
	
	fscpy::PyPromise readyPromise(py::object o) {
		return fscpy::PyPromise(o);
	}
}

namespace fscpy {

void initAsync(py::module_& m) {
	py::class_<PyPromise>(m, "Promise", py::multiple_inheritance(), py::metaclass(*baseMetaType))
		.def(py::init([](PyPromise& other) { return PyPromise(other); }))
		.def("wait", &PyPromise::wait)
		.def("poll", &PyPromise::poll)
		.def("then", &PyPromise::then)
	;
	
	m.def("startEventLoop", &PyContext::startEventLoop);
	m.def("hasEventLoop", &PyContext::hasEventLoop);
	m.def("dataService", &dataService);
	m.def("cycle", &cycle);
	m.def("readyPromise", &readyPromise);
	
	auto atexitModule = py::module_::import("atexit");
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}
	
}