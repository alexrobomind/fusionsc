#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <fsc/common.h>

namespace py = pybind11;

namespace fscpy {
	using namespace fsc;
	
	void defCapnpClasses(py::module_ m);	
}

// Array pointer conversion

namespace pybind11 { namespace detail {
	
template<>
struct type_caster<kj::StringPtr> {
	PYBIND11_TYPE_CASTER(kj::StringPtr, const_name("StringPtr"));
	
	type_caster<char> strCaster;	
	
	bool load(handle src, bool convert) {			
		object isInstance = eval("isinstance");
		
		if(isInstance(src, eval("str"))) {
			if(strCaster.load(src, convert)) {
				auto asCharPtr = (char*) strCaster;
				auto len = strlen(asCharPtr);
				value = kj::StringPtr(asCharPtr, len);
				return true;
			}
		}
		
		if(isInstance(src, type::of<kj::String>())) {
			value = src.cast<kj::String&>().asPtr();
			return true;
		}
		
		return false;
	}
	
	static handle cast(kj::StringPtr src, return_value_policy policy, handle parent) {
		kj::String result = kj::heapString(src);
		return py::cast(kj::mv(result));
	}
};

template<typename T>
struct type_caster<kj::ArrayPtr<T>> {
	PYBIND11_TYPE_CASTER(kj::ArrayPtr<T>, const_name("ArrayPtr"));
	
	bool load(handle src, bool convert) {
		try {
			kj::Array<kj::RemoveConst<T>>& ref = src.cast<kj::Array<kj::RemoveConst<T>>&>();
			value = ref.asPtr();
			return true;
		} catch(cast_error& e) {
		}
		
		try {
			kj::Array<T>& ref = src.cast<kj::Array<T>&>();
			value = ref.asPtr();
			return true;
		} catch(cast_error& e) {
		}
		
		return false;
	}
	
	static handle cast(kj::ArrayPtr<T> src, return_value_policy policy, handle parent) {
		kj::Array<T> result = kj::heapArray(src);
		return py::cast(kj::mv(result));
	}
};

}}