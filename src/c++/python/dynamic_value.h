#pragma once

#include "fscpy.h"

#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <capnp/dynamic.h>
#include <capnp/any.h>

// Conversion of capnp::DynamicValue to python objects
extern pybind11::dict globalBuilderClasses;
extern pybind11::dict globalReaderClasses;

namespace pybind11 { namespace detail {
	
	template<>
	struct type_caster<capnp::DynamicValue::Builder> {
		using DynamicValue = capnp::DynamicValue;
		using DynamicList = capnp::DynamicList;
		using DynamicStruct = capnp::DynamicStruct;
		using DynamicEnum = capnp::DynamicEnum;
		using DynamicCapability = capnp::DynamicCapability;
		using AnyPointer = capnp::AnyPointer;
		
		PYBIND11_TYPE_CASTER(DynamicValue::Builder, const_name("DynamicValueBuilder"));
		
		bool load(handle src, bool convert) {			
			object isInstance = eval("isinstance");
			if(isInstance(src, type::of<DynamicStruct::Builder>())) {
				value = src.cast<DynamicStruct::Builder>();
				return true;
			}
			
			if(isInstance(src, type::of<DynamicList::Builder>())) {
				value = src.cast<DynamicList::Builder>();
				return true;
			}
			
			if(isInstance(src, type::of<DynamicEnum>())) {
				value = src.cast<DynamicEnum>();
				return true;
			}
			
			if(isInstance(src, type::of<DynamicCapability::Client>())) {
				value = src.cast<DynamicCapability::Client>();
				return true;
			}
			
			if(isInstance(src, type::of<AnyPointer::Builder>())) {
				value = src.cast<AnyPointer::Builder>();
				return true;
			}
			
			return false;
		}
		
		static handle cast(DynamicValue::Builder src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::VOID: return none();
				case DynamicValue::BOOL: return py::cast(src.as<bool>());
				case DynamicValue::INT: return py::cast(src.as<signed long long>());
				case DynamicValue::UINT: return py::cast(src.as<unsigned long long>());
				case DynamicValue::FLOAT: return py::cast(src.as<double>());
				case DynamicValue::TEXT: return py::cast(src.as<capnp::Text>());
				case DynamicValue::DATA: return py::cast(src.as<capnp::Data>());
				case DynamicValue::LIST: return py::cast(src.as<capnp::DynamicList>());
				case DynamicValue::ENUM: return py::cast(src.as<capnp::DynamicEnum>());
				case DynamicValue::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>());
				case DynamicValue::ANY_POINTER: return py::cast(src.as<capnp::AnyPointer>());
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT);
			
			DynamicStruct::Builder dynamicStruct = src.as<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalBuilderClasses.contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = globalBuilderClasses[py::cast(typeId)];
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				auto result = targetClass(dynamicStruct);
				return result;
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Builder>::cast((DynamicStruct::Builder &&) src, policy, parent);
		}
	};
	
	template<>
	struct type_caster<capnp::DynamicValue::Reader> {
		using DynamicValue = capnp::DynamicValue;
		using DynamicList = capnp::DynamicList;
		using DynamicStruct = capnp::DynamicStruct;
		using DynamicEnum = capnp::DynamicEnum;
		using DynamicCapability = capnp::DynamicCapability;
		using AnyPointer = capnp::AnyPointer;
		
		PYBIND11_TYPE_CASTER(DynamicValue::Reader, const_name("DynamicValueReader"));
		
		// If we get a string, we need to store it temporarily
		type_caster<char> strCaster;		
		
		bool load(handle src, bool convert) {
			object pyType = eval("type")(src);
			
			if(pyType == eval("real")) {
				value = src.cast<double>();
				return true;
			}
			
			if(pyType == eval("int")) {
				if(src >= eval("0")) {
					value = src.cast<unsigned long long>();
				} else {
					value = src.cast<signed long long>();
				}
				return true;
			}
			
			if(pyType == eval("str")) {
				strCaster.load(src, false);
				value = capnp::Text::Reader((char*) strCaster);
				return true;
			}
			
			object isInstance = eval("isinstance");
			if(isInstance(src, type::of<DynamicStruct::Reader>())) {
				value = src.cast<DynamicStruct::Reader>();
				return true;
			}
			
			if(isInstance(src, type::of<DynamicList::Reader>())) {
				value = src.cast<DynamicList::Reader>();
				return true;
			}
			
			if(isInstance(src, type::of<DynamicEnum>())) {
				value = src.cast<DynamicEnum>();
				return true;
			}
			
			if(isInstance(src, type::of<AnyPointer::Reader>())) {
				value = src.cast<AnyPointer::Reader>();
				return true;
			}
			
			if(isInstance(src, type::of<DynamicCapability::Client>())) {
				value = src.cast<DynamicCapability::Client>();
				return true;
			}
			
			type_caster<DynamicValue::Builder> builderCaster;
			if(builderCaster.load(src, convert)) {
				value = ((DynamicValue::Builder& ) builderCaster).asReader();
				return true;
			}
			
			return false;
		}
		
		
		static handle cast(DynamicValue::Reader src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::VOID: return none();
				case DynamicValue::BOOL: return py::cast(src.as<bool>());
				case DynamicValue::INT: return py::cast(src.as<signed long long>());
				case DynamicValue::UINT: return py::cast(src.as<unsigned long long>());
				case DynamicValue::FLOAT: return py::cast(src.as<double>());
				case DynamicValue::TEXT: return py::cast(src.as<capnp::Text>());
				case DynamicValue::DATA: return py::cast(src.as<capnp::Data>());
				case DynamicValue::LIST: return py::cast(src.as<capnp::DynamicList>());
				case DynamicValue::ENUM: return py::cast(src.as<capnp::DynamicEnum>());
				// case DynamicValue::STRUCT: return py::cast(src.as<capnp::DynamicStruct>());
				case DynamicValue::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>());
				case DynamicValue::ANY_POINTER: return py::cast(src.as<capnp::AnyPointer>());
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT);
			
			DynamicStruct::Reader dynamicStruct = src.as<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalReaderClasses.contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = globalReaderClasses[py::cast(typeId)];
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				auto result = targetClass(dynamicStruct);
				return result;
			}
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Reader>::cast((DynamicStruct::Reader &&) src, policy, parent);
		}
	};
}}