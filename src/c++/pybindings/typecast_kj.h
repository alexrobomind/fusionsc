#pragma once

#include <pybind11/cast.h>
#include <pybind11/eval.h>

#include <kj/array.h>
#include <kj/string.h>

#include "kj.h"

// This file contains the following type caster specializations:
//
//   kj::StringPtr
//   template<typename T> kj::ArrayPtr<T>

namespace fscpy {

template<typename T>
struct DynamicArrayImpl : public DynamicArray {
	kj::ArrayPtr<T> asPtr;
	
	Maybe<kj::Array<T>> keepAlive;
	
	DynamicArrayImpl(kj::Array<T> newArr) :
		asPtr(newArr.asPtr()),
		keepAlive(mv(newArr))
	{}
	
	DynamicArrayImpl(kj::ArrayPtr<T> newArr) :
		asPtr(newArr.asPtr())
	{}
	
	py::object get(size_t i) override {
		return py::cast(asPtr[i]);
	}
	
	void set(size_t i, py::object newVal) override {
		asPtr[i] = py::cast<T>(newVal);
	}
	
	size_t size() override { return asPtr.size(); }
	
	~DynamicArrayImpl() noexcept {};
};

template<typename T>
struct DynamicArrayImpl<const T> : public DynamicConstArray {
	kj::ArrayPtr<const T> asPtr;
	
	Maybe<kj::Array<const T>> keepAlive;
	
	DynamicArrayImpl(kj::Array<const T> newArr) :
		asPtr(newArr.asPtr()),
		keepAlive(mv(newArr))
	{}
	
	DynamicArrayImpl(kj::ArrayPtr<const T> newArr) :
		asPtr(newArr.asPtr())
	{}
	
	py::object get(size_t i) override {
		return py::cast(asPtr[i]);
	}
	
	size_t size() override { return asPtr.size(); }
	
	~DynamicArrayImpl() noexcept {};
};

}

namespace pybind11 { namespace detail {
	
template<>
struct type_caster<kj::StringPtr> {
	
	PYBIND11_TYPE_CASTER(kj::StringPtr, const_name("str"));
	FSCPY_MOVE_ONLY_CASTER;
	
	type_caster<char> strCaster;	
	
	bool load(handle src, bool convert) {			
		object isInstance = eval("isinstance");
		
		if(PyUnicode_Check(src.ptr())) {
			Py_ssize_t bufSize;
			
			const char* buf = PyUnicode_AsUTF8AndSize(src.ptr(), &bufSize);
			if(buf == nullptr)
				throw py::error_already_set();
			
			value = kj::StringPtr(buf, bufSize);
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
		const char* pStr = src.cStr();		
		return type_caster<char>::cast(pStr, policy, parent);
	}
};
	
template<>
struct type_caster<kj::String> {
	
	PYBIND11_TYPE_CASTER(kj::String, const_name("str"));
	FSCPY_MOVE_ONLY_CASTER;
	
	bool load(handle src, bool convert) {			
		object isInstance = eval("isinstance");
		
		type_caster<kj::StringPtr> ptrCaster;
		if(ptrCaster.load(src, convert)) {
			value = kj::heapString((kj::StringPtr) ptrCaster);
			return true;
		}
		
		type_caster_base<kj::String> base;
		if(base.load(src, convert)) {
			value = kj::heapString((kj::String&) base);
			return true;
		}
		
		return false;
	}
	
	static handle cast(const kj::String& src, return_value_policy policy, handle parent) {
		const char* pStr = src.cStr();		
		return type_caster<char>::cast(pStr, policy, parent);
	}
};

template<typename T>
struct NameForArray { static constexpr auto name = const_name("Mutable array of ") + const_name<T>(); };

template<typename T>
struct NameForArray<const T> { static constexpr auto name = const_name("Array of ") + const_name<T>(); };

template<>
struct NameForArray<unsigned char> { static inline constexpr auto name = const_name("Mutable Bytes"); };

template<>
struct NameForArray<const unsigned char> { static inline constexpr auto name = const_name("Bytes"); };

/*template<typename T>
struct type_caster<kj::ArrayPtr<T>> {
	
	PYBIND11_TYPE_CASTER(kj::ArrayPtr<T>, NameForArray<T>::name);
	
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
		return type_caster<kj::Array<T>>::cast(kj::mv(result), policy, parent);
	}
};*/

template<typename T>
struct type_caster<kj::ArrayPtr<T>> {
	PYBIND11_TYPE_CASTER(kj::ArrayPtr<T>, NameForArray<T>::name);
	FSCPY_MOVE_ONLY_CASTER;
	
	bool load(handle src, bool convert) {
		// Check if it is an array we returned
		type_caster<fscpy::DynamicArray> base;
		if(!base.load(src, convert))
			return false;
		
		// Check if the array is of correct type
		fscpy::DynamicArrayImpl<T>* ptr = dynamic_cast<fscpy::DynamicArrayImpl<T>*>((fscpy::DynamicArray*) base);
		if(ptr) {
			value = ptr->asPtr;
			return true;
		}
		
		return false;
	}
	
	static handle cast(kj::ArrayPtr<T> src, return_value_policy policy, handle parent) {
		return type_caster<fscpy::DynamicArray>::cast(fscpy::DynamicArray(kj::heapArray(src)), policy, parent);
	}
};

template<typename T>
struct type_caster<kj::Array<T>> {
	PYBIND11_TYPE_CASTER(kj::Array<T>, NameForArray<T>::name);
	FSCPY_MOVE_ONLY_CASTER;
	
	// Arrays are owning objects. They can not be passed in via the
	// standard pybind11 mechanisms.
	bool load(handle src, bool convert) {
		return false;
	}
	
	static handle cast(kj::Array<T> src, return_value_policy policy, handle parent) {
		return type_caster<fscpy::DynamicArray>::cast(fscpy::DynamicArray(kj::mv(src)), policy, parent);
	}
};

template<typename T>
struct type_caster<kj::ArrayPtr<const T>> {
	PYBIND11_TYPE_CASTER(kj::ArrayPtr<T>, NameForArray<T>::name);
	FSCPY_MOVE_ONLY_CASTER;
	
	bool load(handle src, bool convert) {
		// Check if it is an array we returned
		type_caster<fscpy::DynamicArray> base;
		if(!base.load(src, convert))
			return false;
		
		// Check if the array is of correct type
		{
			// Attempt 1: Mutable array
			fscpy::DynamicArrayImpl<T>* ptr = dynamic_cast<fscpy::DynamicArrayImpl<T>*>((fscpy::DynamicArray*) base);
			if(ptr) {
				value = ptr->asPtr;
				return true;
			}
		}
		{
			// Attempt 2: Const array
			fscpy::DynamicArrayImpl<const T>* ptr = dynamic_cast<fscpy::DynamicArrayImpl<const T>*>((fscpy::DynamicArray*) base);
			if(ptr) {
				value = ptr->asPtr;
				return true;
			}
		}
		
		return false;
	}
	
	static handle cast(kj::ArrayPtr<const T> src, return_value_policy policy, handle parent) {
		return type_caster<fscpy::DynamicConstArray>::cast(fscpy::DynamicArrayImpl(kj::heapArray(src)), policy, parent);
	}
};

template<typename T>
struct type_caster<kj::Array<const T>> {
	PYBIND11_TYPE_CASTER(kj::Array<T>, NameForArray<T>::name);
	FSCPY_MOVE_ONLY_CASTER;
	
	// Arrays are owning objects. They can not be passed in via the
	// standard pybind11 mechanisms.
	bool load(handle src, bool convert) {
		return false;
	}
	
	static handle cast(kj::Array<const T> src, return_value_policy policy, handle parent) {
		return type_caster<fscpy::DynamicConstArray>::cast(fscpy::DynamicArrayImpl(kj::mv(src)), policy, parent);
	}
};

}}