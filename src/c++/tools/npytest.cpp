#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace py = pybind11;

int main() {
	py::scoped_interpreter interpreter{};
	// py::module_::import("numpy");
	import_array();
	
	PyArray_Descr* descr1 = PyArray_DescrFromType(NPY_UINT8);	
	PyArray_Descr* descr2 = PyArray_DescrNewByteorder(descr1, NPY_LITTLE);
	
	PyArray_RegisterDataType(descr2);
	if(PyErr_Occurred()) {
		throw py::error_already_set();
	}
	
	py::print(descr1 -> type_num, descr2 -> type_num);
	return 0;
}