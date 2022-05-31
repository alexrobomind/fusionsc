#include "async.h"

namespace fscpy {

void bindAsyncClasses(py::module_& m) {
	py::class_<PyPromise>(m, "Promise")
		.def(py::init([](PyPromise& other) { return PyPromise(other); }))
	;
}
	
}