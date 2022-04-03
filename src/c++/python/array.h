#pragma once

#include <kj/array.h>
#include <kj/string.h>

// This file contains the following type caster specializations:
//
//   kj::StringPtr
//   template<typename T> kj::ArrayPtr<T>

// Array pointer conversion
namespace pybind11 { namespace detail {
	
template<>
struct type_caster<kj::StringPtr> {
	PYBIND11_TYPE_CASTER(kj::StringPtr, const_name("str"));
	
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
		
		type_caster_base<kj::StringPtr> base;
		if(base.load(src, convert)) {
			value = (kj::StringPtr) base;
			return true;
		}
		
		return false;
	}
	
	static handle cast(kj::StringPtr src, return_value_policy policy, handle parent) {
		/*kj::String result = kj::heapString(src);
		return py::cast(kj::mv(result));*/
		const char* pStr = src.begin();
		return py::cast(pStr);
	}
};

template<typename T>
struct type_caster<kj::ArrayPtr<T>> {
	template<typename T>
	struct NameFor_ { static constexpr auto name = const_name("Mutable array of ") + const_name<T>(); };

	template<typename T>
	struct NameFor_<const T> { static constexpr auto name = const_name("Array of ") + const_name<T>(); };

	template<>
	struct NameFor_<unsigned char> { static inline constexpr auto name = const_name("Mutable Bytes"); };

	template<>
	struct NameFor_<const unsigned char> { static constexpr auto name = const_name("Bytes"); };
	
	PYBIND11_TYPE_CASTER(kj::ArrayPtr<T>, NameFor_<T>::name);
	
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
		
		type_caster_base<kj::ArrayPtr<T>> base;
		if(base.load(src, convert)) {
			value = (kj::ArrayPtr<T>) base;
			return true;
		}
		
		return false;
	}
	
	static handle cast(kj::ArrayPtr<T> src, return_value_policy policy, handle parent) {
		kj::Array<T> result = kj::heapArray(src);
		return py::cast(kj::mv(result));
	}
};

}}