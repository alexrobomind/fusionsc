#pragma once

#include "fscpy.h"

#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <capnp/dynamic.h>
#include <capnp/any.h>

namespace pybind11 { namespace detail {
	
	template<>
	struct type_caster<capnp::DynamicValue::Builder> {
		using DynamicValue = capnp::DynamicValue;
		using DynamicList = capnp::DynamicList;
		using DynamicStruct = capnp::DynamicStruct;
		using DynamicEnum = capnp::DynamicEnum;
		using DynamicCapability = capnp::DynamicCapability;
		using AnyPointer = capnp::AnyPointer;
		
		PYBIND11_TYPE_CASTER(DynamicValue::Builder, const_name("{Dynamic builder or primitive value}"));
		
		bool load(handle src, bool convert) {			
			#define FSCPY_TRY_CAST(Type) \
				try { \
					value = src.cast<Type>(); \
					return true; \
				} catch(cast_error& e) { \
				}
			
			FSCPY_TRY_CAST(DynamicStruct::Builder)
			FSCPY_TRY_CAST(DynamicList::Builder)
			FSCPY_TRY_CAST(DynamicEnum)
			FSCPY_TRY_CAST(DynamicCapability::Client)
			FSCPY_TRY_CAST(AnyPointer::Builder)
			
			#undef FSCPY_TRY_CAST
			
			type_caster_base<DynamicValue::Builder> base;
			auto baseLoadResult = base.load(src, convert);
			
			if(baseLoadResult) {
				value = (DynamicValue::Builder) base;
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
			
			if(globalClasses->contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Builder");
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				auto result = targetClass(dynamicStruct);
				return result;
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Builder>::cast(dynamicStruct, policy, parent);
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
		
		PYBIND11_TYPE_CASTER(DynamicValue::Reader, const_name("{Dynamic reader, builder or primitive value}"));
		
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
			
			#define FSCPY_TRY_CAST(Type) \
				try { \
					value = src.cast<Type>(); \
					return true; \
				} catch(cast_error& e) { \
				}
			
			FSCPY_TRY_CAST(DynamicStruct::Reader)
			FSCPY_TRY_CAST(DynamicList::Reader)
			FSCPY_TRY_CAST(DynamicEnum)
			FSCPY_TRY_CAST(DynamicCapability::Client)
			
			#undef FSCPY_TRY_CAST
			
			try { 
				value = src.cast<DynamicValue::Builder>().asReader();
				return true; 
			} catch(cast_error& e) { 
			}
			
			type_caster_base<DynamicValue::Reader> base;
			auto baseLoadResult = base.load(src, convert);
			
			if(baseLoadResult) {
				value = (DynamicValue::Reader) base;
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
				case DynamicValue::ANY_POINTER: return py::cast(src.as<capnp::AnyPointer>());
				case DynamicValue::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>());
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT);
			
			DynamicStruct::Reader dynamicStruct = src.as<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalClasses->contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Reader");
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				auto result = targetClass(dynamicStruct);
				return result;
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Reader>::cast(dynamicStruct, policy, parent);
		}
	};
	
	template<>
	struct type_caster<capnp::DynamicValue::Pipeline> {
		using DynamicValue = capnp::DynamicValue;
		using DynamicList = capnp::DynamicList;
		using DynamicStruct = capnp::DynamicStruct;
		using DynamicEnum = capnp::DynamicEnum;
		using DynamicCapability = capnp::DynamicCapability;
		using AnyPointer = capnp::AnyPointer;
		
		PYBIND11_TYPE_CASTER(DynamicValue::Pipeline, const_name("{Dynamic pipeline}"));
		
		bool load(handle src, bool convert) {
			return false;
		}
		
		static handle cast(DynamicValue::Pipeline src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::CAPABILITY: return py::cast(src.releaseAs<capnp::DynamicCapability>());
				// case DynamicValue::ANY_POINTER: return py::cast(src.releaseAs<capnp::AnyPointer>());
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT);
			
			DynamicStruct::Pipeline dynamicStruct = src.releaseAs<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalClasses->contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Pipeline");
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				auto result = targetClass(dynamicStruct);
				return result;
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Pipeline>::cast(dynamicStruct, policy, parent);
		}
	};
	
	template<>
	struct type_caster<capnp::DynamicCapability::Client> {
		using DynamicCapability = capnp::DynamicCapability;
		
		PYBIND11_TYPE_CASTER(capnp::DynamicCapability::Client, const_name("DynamicCapability.Client"));
		
		bool load(handle src, bool convert) {
			type_caster_base<DynamicCapability::Client> base;
			if(base.load(src, convert)) {
				value = (DynamicCapability::Client) base;
				return true;
			}
			
			return false;		
		}
		
		static handle cast(DynamicCapability::Client src, return_value_policy policy, handle parent) {
			auto typeId = src.getSchema().getProto().getId();
			
			if(globalClasses->contains(typeId)) {
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Client");
				
				auto result = targetClass(src);
				return result;
			}
			
			return type_caster_base<DynamicCapability::Client>::cast(src, policy, parent);
		}
	};
}}