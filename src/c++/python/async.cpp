#include "async.h"

#include <capnp/dynamic.h>

#include <fsc/data.h>

namespace {
	fsc::LocalDataService& dataService()  {
		return fscpy::PyContext::libraryThread()->dataService();
	}
}

namespace fscpy {

void bindAsyncClasses(py::module_& m) {
	py::class_<PyPromise>(m, "Promise")
		.def(py::init([](PyPromise& other) { return PyPromise(other); }))
	;
	
	m.def("startEventLoop", &PyContext::startEventLoop);
	m.def("hasEventLoop", &PyContext::hasEventLoop);
	m.def("dataService", &dataService);
}
	
}