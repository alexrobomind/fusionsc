#pragma once

#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <capnp/dynamic.h>
#include <capnp/any.h>
#include <kj/common.h>

#include "loader.h"
#include "capnp.h"

// This file contains the following type caster specializations:
//
//   capnp::DynamicValue::Builder
//   capnp::DynamicValue::Reader
//   capnp::DynamicValue::Pipeline
//   capnp::DynamicCapability::Client

namespace pybind11 { namespace detail {
	
	template<>
	struct type_caster<capnp::DynamicCapability::Client> {
		using DynamicCapability = capnp::DynamicCapability;
		
		PYBIND11_TYPE_CASTER(capnp::DynamicCapability::Client, const_name("DynamicCapability.Client"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() = default;
		type_caster(const type_caster<capnp::DynamicCapability::Client>& other) = delete;
		type_caster(type_caster<capnp::DynamicCapability::Client>&& other) = default;
		
		
		bool load(handle src, bool convert) {
			type_caster_base<DynamicCapability::Client> base;
			if(base.load(src, convert)) {
				value = base;
				return true;
			}
			
			return false;		
		}
		
		static handle cast(capnp::DynamicCapability::Client src, return_value_policy policy, handle parent) {
			auto typeId = src.getSchema().getProto().getId();
			
			// Special handling for normal Capability::Client
			KJ_IF_MAYBE(pythonSchema, ::fscpy::defaultLoader.capnpLoader.tryGet(typeId)) {
				KJ_REQUIRE(pythonSchema->getProto().isInterface());
				
				if(*pythonSchema != src.getSchema().getGeneric()) {
					KJ_REQUIRE(!src.getSchema().isBranded(), "Can only pass types without default parametrization from C++ to python");
					
					try {						
						capnp::Capability::Client untyped = src;
						src = untyped.castAs<capnp::DynamicCapability>(pythonSchema->asInterface());
						
						KJ_REQUIRE(src.getSchema() == ::fscpy::defaultLoader.capnpLoader.get(typeId));
					} catch(kj::Exception& e) {
						KJ_FAIL_REQUIRE("Failed to read python schema", typeId, src.getSchema().getProto().getDisplayName());
					}
				}
			}
			
			object baseClient = reinterpret_steal<object>(type_caster_base<capnp::DynamicCapability::Client>::cast(kj::mv(src), policy, parent));
			
			if(globalClasses->contains(typeId)) {
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Client");
				
				object result = targetClass(baseClient);
				return result.inc_ref();
			}
			
			return baseClient.inc_ref();
		}
	};
	
	template<>
	struct type_caster<capnp::DynamicValue::Builder> {
		using DynamicValue = capnp::DynamicValue;
		using DynamicList = capnp::DynamicList;
		using DynamicStruct = capnp::DynamicStruct;
		using DynamicEnum = capnp::DynamicEnum;
		using DynamicCapability = capnp::DynamicCapability;
		using AnyPointer = capnp::AnyPointer;
		
		PYBIND11_TYPE_CASTER(DynamicValue::Builder, const_name("DynamicValue.Builder"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() = default;
		type_caster(const type_caster<capnp::DynamicValue::Builder>& other) = delete;
		type_caster(type_caster<capnp::DynamicValue::Builder>&& other) = default;
		
		bool load(handle src, bool convert) {			
			#define FSCPY_TRY_CAST(Type) \
				{ \
					type_caster<Type> caster; \
					if(caster.load(src, convert)) { \
						value = (Type&) caster; \
						return true; \
					} \
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
				value = (DynamicValue::Builder&) base;
				return true;
			}
			
			return false;
		}
		
		static handle cast(DynamicValue::Builder src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::VOID: return none().inc_ref();
				case DynamicValue::BOOL: return py::cast(src.as<bool>()).inc_ref();
				case DynamicValue::INT: return py::cast(src.as<int64_t>()).inc_ref();
				case DynamicValue::UINT: return py::cast(src.as<uint64_t>()).inc_ref();
				case DynamicValue::FLOAT: return py::cast(src.as<double>()).inc_ref();
				case DynamicValue::TEXT: return py::cast(src.as<capnp::Text>()).inc_ref();
				case DynamicValue::DATA: return py::cast(src.as<capnp::Data>()).inc_ref();
				case DynamicValue::LIST: return py::cast(src.as<capnp::DynamicList>()).inc_ref();
				case DynamicValue::ENUM: return py::cast(src.as<capnp::DynamicEnum>()).inc_ref();
				case DynamicValue::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>()).inc_ref();
				case DynamicValue::ANY_POINTER: return py::cast(src.as<capnp::AnyPointer>()).inc_ref();
				default: {}
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT, "Unknown input type");
			
			DynamicStruct::Builder dynamicStruct = src.as<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalClasses->contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Builder");
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				object result = targetClass(dynamicStruct);
				return result.inc_ref();
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
		
		PYBIND11_TYPE_CASTER(DynamicValue::Reader, const_name("DynamicValue.Reader"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() = default;
		type_caster(const type_caster<capnp::DynamicValue::Reader>& other) = delete;
		type_caster(type_caster<capnp::DynamicValue::Reader>&& other) = default;
		
		// If we get a string, we need to store it temporarily
		type_caster<char> strCaster;		
		
		bool load(handle src, bool convert) {
			object pyType = eval("type")(src);
						
			if(src.is_none()) {
				value = capnp::Void();
				return true;
			}
			
			if(pyType.equal(eval("float"))) {
				value = src.cast<double>();
				return true;
			}
			
			if(pyType.equal(eval("int"))) {
				if(src >= eval("0")) {
					value = src.cast<unsigned long long>();
				} else {
					value = src.cast<signed long long>();
				}
				return true;
			}
			
			if(pyType.equal(eval("str"))) {
				strCaster.load(src, false);
				value = capnp::Text::Reader((char*) strCaster);
				return true;
			}
						
			#define FSCPY_TRY_CAST(Type) \
				{ \
					type_caster<Type> caster; \
					if(caster.load(src, convert)) { \
						value = (Type&) caster; \
						return true; \
					} \
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
				value = (DynamicValue::Reader&) base;
				return true;
			}
			
			return false;
		}
		
		
		static handle cast(DynamicValue::Reader src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::VOID: return none();
				case DynamicValue::BOOL: return py::cast(src.as<bool>()).inc_ref();
				case DynamicValue::INT: return py::cast(src.as<int64_t>()).inc_ref();
				case DynamicValue::UINT: return py::cast(src.as<uint64_t>()).inc_ref();
				case DynamicValue::FLOAT: return py::cast(src.as<double>()).inc_ref();
				case DynamicValue::TEXT: return py::cast(src.as<capnp::Text>()).inc_ref();
				case DynamicValue::DATA: return py::cast(src.as<capnp::Data>()).inc_ref();
				case DynamicValue::LIST: return py::cast(src.as<capnp::DynamicList>()).inc_ref();
				case DynamicValue::ENUM: return py::cast(src.as<capnp::DynamicEnum>()).inc_ref();
				// case DynamicValue::STRUCT: return py::cast(src.as<capnp::DynamicStruct>());
				case DynamicValue::ANY_POINTER: return py::cast(src.as<capnp::AnyPointer>()).inc_ref();
				case DynamicValue::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>()).inc_ref();
				default: {}
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT, "Unknown input type");
			
			DynamicStruct::Reader dynamicStruct = src.as<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalClasses->contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Reader");
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				object result = targetClass(dynamicStruct);
				return result.inc_ref();
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Reader>::cast(dynamicStruct, policy, parent);
		}
	};
	
	template<>
	struct type_caster<fscpy::DynamicValuePipeline> {
		using DynamicValuePipeline = fscpy::DynamicValuePipeline;
		using DynamicStructPipeline = fscpy::DynamicStructPipeline;
		
		PYBIND11_TYPE_CASTER(DynamicValuePipeline, const_name("DynamicValuePipeline"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() = default;
		type_caster(const type_caster<DynamicValuePipeline>& other) = delete;
		type_caster(type_caster<DynamicValuePipeline>&& other) = default;
		
		// Currently we don't have a use case to pass pipelines into C++ code
		// This might change in the future
		bool load(handle src, bool convert) {
			return false;
		}
		
		static handle cast(DynamicValuePipeline src, return_value_policy policy, handle parent) {
			if(src.schema.getProto().isInterface()) {
				return type_caster<capnp::DynamicCapability::Client>::cast(src.asCapability(), policy, parent);
			}
			
			DynamicStructPipeline dynamicStruct = src.asStruct();
			auto typeId = dynamicStruct.schema.getProto().getId();
			
			if(globalClasses->contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Pipeline");
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Pipeline, which has a
				// copy initializer. The target object will be forwarded to it
				object result = targetClass(dynamicStruct);
				return result.inc_ref();
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStructPipeline>::cast(dynamicStruct, policy, parent);
		}
	};
	
	/*
	template<>
	struct type_caster<capnp::DynamicValue::Pipeline> {
		using DynamicValue = capnp::DynamicValue;
		using DynamicList = capnp::DynamicList;
		using DynamicStruct = capnp::DynamicStruct;
		using DynamicEnum = capnp::DynamicEnum;
		using DynamicCapability = capnp::DynamicCapability;
		using AnyPointer = capnp::AnyPointer;
		
		PYBIND11_TYPE_CASTER(DynamicValue::Pipeline, const_name("DynamicValue.Pipeline"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() = default;
		type_caster(const type_caster<capnp::DynamicValue::Pipeline>& other) = delete;
		type_caster(type_caster<capnp::DynamicValue::Pipeline>&& other) = default;
		
		bool load(handle src, bool convert) {
			return false;
		}
		
		static handle cast(DynamicValue::Pipeline src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::CAPABILITY: return type_caster<capnp::DynamicCapability::Client>::cast(src.releaseAs<capnp::DynamicCapability>(), policy, parent);
				// case DynamicValue::ANY_POINTER: return py::cast(src.releaseAs<capnp::AnyPointer>());
				default: {}
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT);
			
			DynamicStruct::Pipeline dynamicStruct = src.releaseAs<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalClasses->contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Pipeline");
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Pipeline, which has a
				// copy initializer. The target object will be forwarded to it
				object result = targetClass(dynamicStruct, ::fscpy::INTERNAL_ACCESS_KEY);
				return result.inc_ref();
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Pipeline>::cast(dynamicStruct, policy, parent);
		}
	};*/
	
	/*template<>
	struct type_caster<capnp::RemotePromise<capnp::DynamicStruct>> {
		using DynamicValue = capnp::DynamicValue;
		using DynamicList = capnp::DynamicList;
		using DynamicStruct = capnp::DynamicStruct;
		using DynamicEnum = capnp::DynamicEnum;
		using DynamicCapability = capnp::DynamicCapability;
		using AnyPointer = capnp::AnyPointer;
		
		template<typename T>
		using RemotePromise = capnp::RemotePromise<T>;
		
		template<typename T>
		using Promise = kj::Promise<T>;
		
		PYBIND11_TYPE_CASTER(capnp::RemotePromise<capnp::DynamicStruct>, const_name("DynamicStruct.Remote"));
		
		type_caster() = default;
		type_caster(const type_caster<capnp::RemotePromise<capnp::DynamicStruct>>& other) = delete;
		type_caster(type_caster<capnp::RemotePromise<capnp::DynamicStruct>>&& other) = default;
		
		bool load(handle src, bool convert) {
			return false;
		}
		
		static handle cast(RemotePromise<DynamicStruct> src, return_value_policy policy, handle parent) {
			Promise<DynamicStruct> promisePart(kj::mv(in));
			DynamicStruct::Pipeline pipelinePart(kj::mv(in));
			
			
		}
		
		template<typename T>
		RemotePromise<T> castPromise(RemotePromise<DynamicStruct> in) {
			
			
		}
	};*/
}}