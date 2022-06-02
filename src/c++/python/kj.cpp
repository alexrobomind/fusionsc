#include "fscpy.h"

#include <kj/common.h>

using namespace fscpy;

namespace {

template<typename T>
struct ArraySetDefinition {
	template<typename T2>
	static void def(T2& pyClass) {
		pyClass.def("__setitem__", [](Array<T>& arr, size_t i, T newVal) { arr[i] = newVal; });
	}
};

template<typename T>
struct ArraySetDefinition<const T> {
	template<typename... T2>
	static void def(T2&...) {}
};

template<typename T>
struct ArrayPtrSetDefinition {
	template<typename T2>
	static void def(T2& pyClass) {
		pyClass.def("__setitem__", [](ArrayPtr<T>& arr, size_t i, T newVal) { arr[i] = newVal; });
	}
};

template<typename T>
struct ArrayPtrSetDefinition<const T> {
	template<typename... T2>
	static void def(T2&...) {}
};

template<typename T>
void defArray(kj::StringPtr name, py::module_ m) {	
	py::class_<kj::Array<T>> pyClass(m, name.cStr());
	py::class_<kj::ArrayPtr<T>> pyClass2(m, kj::str(name, "Ptr").cStr());
	
	pyClass.def("__getitem__", [](Array<T>& arr, size_t i) -> T { return arr[i]; });
	pyClass2.def("__getitem__", [](ArrayPtr<T>& arr, size_t i) -> T { return arr[i]; });
	
	ArraySetDefinition<T>::def(pyClass);
	ArrayPtrSetDefinition<T>::def(pyClass2);
}

}

namespace fscpy {
	void bindKJClasses(py::module_& m) {
		py::module_ mkj = m.def_submodule("kj");
		
		defArray<kj::byte>("ByteArray", mkj);
		defArray<const kj::byte>("ConstByteArray", mkj);
		
		py::class_<kj::StringPtr>(mkj, "StringPtr")
			.def("__str__", [](kj::StringPtr ptr) { return ptr.cStr(); })
			.def("__repr__", [](kj::StringPtr ptr) { return ptr.cStr(); })
		;
	}
}