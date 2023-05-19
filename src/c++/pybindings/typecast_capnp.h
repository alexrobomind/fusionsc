#pragma once

#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <capnp/dynamic.h>
#include <capnp/any.h>
#include <kj/common.h>

#include <fsc/typing.h>
#include <fsc/data.h>

#include "loader.h"
#include "capnp.h"

// This file contains the following type caster specializations:
//
//   capnp::DynamicValue::Builder
//   capnp::DynamicValue::Reader
//   capnp::DynamicCapability::Client
//
//   capnp struct readers
//   capnp struct builders
//   capnp capabilities
//
//   fsc::Temporary<...>
//
//   fscpy::DynamicValuePipeline
//   fscpy::DynamicStructPipeline

namespace pybind11 { namespace detail {
	
	// In def_buffer passes the caster directly as argument to the buffer generator. For DynamicValue::{Reader, Builder},
	// we need direct access to the underlying py::object. Therefore, we expand the type caster for these two classes
	// to capture the target object upon loading.
	
	#define FSC_BACKREFERENCING_CASTER(T) \
		template <> \
		struct type_caster<T> : public type_caster_base<T> { \
			py::object storedObject; \
			bool load(handle src, bool convert) { \
				bool ok = type_caster_base<T>::load(src, convert); \
				\
				if(ok) { \
					storedObject = reinterpret_borrow<object>(src); \
				}\
				return ok; \
			} \
			\
			operator object() { return storedObject; } \
		};
	
	FSC_BACKREFERENCING_CASTER(capnp::DynamicStruct::Reader);
	FSC_BACKREFERENCING_CASTER(capnp::DynamicStruct::Builder);
		
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
			KJ_DBG(src.getSchema().getProto(), typeId);
			
			// Special handling for normal Capability::Client
			/*KJ_IF_MAYBE(pythonSchema, ::fscpy::defaultLoader.capnpLoader.tryGet(typeId)) {
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
			}*/
			
			object baseClient = reinterpret_steal<object>(type_caster_base<capnp::DynamicCapability::Client>::cast(kj::mv(src), policy, parent));
			
			if(globalClasses->contains(typeId)) {
				auto targetClass = (*globalClasses)[py::cast(typeId)].attr("Client");
				
				object result = targetClass(baseClient);
				return result.inc_ref();
			}
			
			KJ_DBG("Returning untyped capability");
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
			if(src.is_none()) {
				value = capnp::Void();
				return true;
			}
			
			KJ_IF_MAYBE(pReader, fscpy::dynamicValueFromScalar(src)) {
				value = *pReader;
				return true;
			}
			
			if(py::isinstance<py::str>(src)) {
				strCaster.load(src, false);
				value = capnp::Text::Reader((char*) strCaster);
				return true;
			}
			
			if(py::isinstance<py::bytes>(src)) {
				// strCaster.load(src, false);
				// value = capnp::Text::Reader((char*) strCaster);
				char *buffer = nullptr;
				ssize_t length = 0;
				auto asBytes = py::reinterpret_borrow<py::bytes>(src);
				
				if(PyBytes_AsStringAndSize(asBytes.ptr(), &buffer, &length) != 0)
					throw py::error_already_set();
				
				value = capnp::Data::Reader((const unsigned char*) buffer, (size_t) length);
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
			
			type_caster<DynamicValue::Builder> builderCaster;
			if(builderCaster.load(src, convert)) {
				value = static_cast<DynamicValue::Builder&>(builderCaster).asReader();
				return true;
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
				object result = targetClass(kj::mv(dynamicStruct));
				return result.inc_ref();
			}
			
			KJ_FAIL_REQUIRE("Unknown class type");
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStructPipeline>::cast(kj::mv(dynamicStruct), policy, parent);
		}
	};
		
	template<typename Builder>
	struct type_caster<Builder, typename kj::EnableIf<CAPNP_KIND(capnp::FromBuilder<Builder>) == capnp::Kind::STRUCT && !fsc::isTemporary<Builder>()>> {
		using DynamicStruct = capnp::DynamicStruct;
		using Builds = typename Builder::Builds;
		
		PYBIND11_TYPE_CASTER(Builder, const_name<Builds>() + const_name(".Builder"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() : value(nullptr) {};
		type_caster(const type_caster<Builder, typename kj::EnableIf<CAPNP_KIND(capnp::FromBuilder<Builder>) == capnp::Kind::STRUCT && !fsc::isTemporary<Builder>()>>& other) = delete;
		type_caster(type_caster<Builder, typename kj::EnableIf<CAPNP_KIND(capnp::FromBuilder<Builder>) == capnp::Kind::STRUCT && !fsc::isTemporary<Builder>()>>&& other) = default;
		
		bool load(handle src, bool convert) {
			// Try to load as dynamic struct
			type_caster<DynamicStruct::Builder> subCaster;
			if(!subCaster.load(src, convert))
				return false;
			
			try {
				DynamicStruct::Builder dynamic = (DynamicStruct::Builder&) subCaster;
				capnp::StructSchema staticSchema = fscpy::defaultLoader.importBuiltin<Builds>().asStruct();
				
				KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				capnp::AnyStruct::Builder any = dynamic;
			
				value = any.as<Builds>();
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(Builder src, return_value_policy policy, handle parent) {
			capnp::StructSchema schema = fscpy::defaultLoader.importBuiltin<Builds>().asStruct();
			
			capnp::AnyStruct::Builder any = capnp::toAny(src);
			capnp::DynamicValue::Builder dynamic = any.as<capnp::DynamicStruct>(schema);
			
			return type_caster<capnp::DynamicValue::Builder>::cast(dynamic, policy, parent);
		}		
	};
	
	template<typename Reader>
	struct type_caster<Reader, kj::EnableIf<CAPNP_KIND(capnp::FromReader<Reader>) == capnp::Kind::STRUCT>> {
		using DynamicStruct = capnp::DynamicStruct;
		using Reads = typename Reader::Reads;
		
		PYBIND11_TYPE_CASTER(Reader, const_name<Reads>() + const_name(".Reader"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() = default;
		type_caster(const type_caster<Reader, kj::EnableIf<CAPNP_KIND(capnp::FromReader<Reader>) == capnp::Kind::STRUCT>>& other) = delete;
		type_caster(type_caster<Reader, kj::EnableIf<CAPNP_KIND(capnp::FromReader<Reader>) == capnp::Kind::STRUCT>>&& other) = default;
		
		bool load(handle src, bool convert) {
			// Try to load as builder
			type_caster<typename Reads::Builder> builderCaster;
			if(builderCaster.load(src, convert)) {
				value = ((typename Reads::Builder) builderCaster).asReader();
				return true;
			}
			
			// Try to load as dynamic struct
			type_caster<DynamicStruct::Reader> subCaster;
			if(!subCaster.load(src, convert))
				return false;
			
			try {
				DynamicStruct::Reader dynamic = (DynamicStruct::Reader&) subCaster;
				capnp::StructSchema staticSchema = fscpy::defaultLoader.importBuiltin<Reads>().asStruct();
				
				KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				capnp::AnyStruct::Reader any = dynamic;
			
				value = any.as<Reads>();
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(Reader src, return_value_policy policy, handle parent) {
			capnp::StructSchema schema = fscpy::defaultLoader.importBuiltin<Reads>().asStruct();
			
			capnp::AnyStruct::Reader any = capnp::toAny(src);
			capnp::DynamicValue::Reader dynamic = any.as<capnp::DynamicStruct>(schema);
			
			return type_caster<capnp::DynamicValue::Reader>::cast(dynamic, policy, parent);
		}		
	};
	
	template<typename T>
	struct type_caster<fsc::Temporary<T>, kj::EnableIf<capnp::kind<T>() == capnp::Kind::STRUCT>> {
		PYBIND11_TYPE_CASTER(fsc::Temporary<T>, const_name<T>() + const_name(".Builder"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() = default;
		type_caster(const type_caster<fsc::Temporary<T>, kj::EnableIf<capnp::kind<T>() == capnp::Kind::STRUCT>>& other) = delete;
		type_caster(type_caster<fsc::Temporary<T>, kj::EnableIf<capnp::kind<T>() == capnp::Kind::STRUCT>>&& other) = default;
		
		bool load(handle src, bool convert) {
			return false;
		}
		
		static handle cast(fsc::Temporary<T> src, return_value_policy policy, handle parent) {
			capnp::StructSchema schema = fscpy::defaultLoader.importBuiltin<T>().asStruct();
			
			capnp::AnyStruct::Builder any = capnp::toAny(src);
			capnp::DynamicValue::Builder dynamic = any.as<capnp::DynamicStruct>(schema);
			
			py::handle builder = type_caster<capnp::DynamicValue::Builder>::cast(dynamic, policy, parent);
			
			auto holder = new fscpy::UnknownHolder<kj::Own<capnp::MessageBuilder>>(mv(src.holder));
			py::object msg = py::cast((fscpy::UnknownObject*) holder);
			builder.attr("_msg") = msg;
			
			return builder;
		}		
	};
	
	
	namespace pybind_fsc {
		template<typename T>
		struct CastThisCap_ { static constexpr bool val = capnp::kind<T>() == capnp::Kind::INTERFACE; };
		
		template<>
		struct CastThisCap_<capnp::DynamicCapability> { static constexpr bool val = false; };
		
		template<typename T>
		constexpr bool castThisCap() { return CastThisCap_<T>::val; }		
	}
	
	template<typename Client>
	struct type_caster<Client, kj::EnableIf<pybind_fsc::castThisCap<capnp::FromClient<Client>>()>> {	
		using ClientFor = capnp::FromClient<Client>;
		
		PYBIND11_TYPE_CASTER(Client, const_name<ClientFor>() + const_name(".Client"));
		
		// We need this so libstdc++ can declare tuples involving this class
		type_caster() : value(nullptr) {};
		type_caster(const type_caster<Client, kj::EnableIf<pybind_fsc::castThisCap<capnp::FromClient<Client>>()>>& other) = delete;
		type_caster(type_caster<Client, kj::EnableIf<pybind_fsc::castThisCap<capnp::FromClient<Client>>()>>&& other) = default;
		
		bool load(handle src, bool convert) {
			// Try to load as dynamic struct
			type_caster<capnp::DynamicCapability::Client> subCaster;
			if(!subCaster.load(src, convert)) {
				return false;
			}
			
			try {
				capnp::DynamicCapability::Client dynamic = (capnp::DynamicCapability::Client&) subCaster;
				
				capnp::InterfaceSchema staticSchema = fscpy::defaultLoader.importBuiltin<ClientFor>().asInterface();
				
				if(dynamic.getSchema() != staticSchema) {
					fsc::Temporary<capnp::schema::Type> t1;
					fsc::Temporary<capnp::schema::Type> t2;
					
					fsc::extractType(dynamic.getSchema(), t1.asBuilder());
					fsc::extractType(staticSchema, t2.asBuilder());
					
					KJ_DBG(t1.asReader());
					KJ_DBG(t2.asReader());
				}
				
				KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				capnp::Capability::Client any = dynamic;
			
				value = dynamic.castAs<ClientFor>();
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(Client src, return_value_policy policy, handle parent) {
			capnp::InterfaceSchema schema = fscpy::defaultLoader.importBuiltin<ClientFor>().asInterface();
			
			capnp::Capability::Client anyCap = src;
			capnp::DynamicCapability::Client dynamic = src.template castAs<capnp::DynamicCapability>(schema);
			
			return type_caster<capnp::DynamicCapability::Client>::cast(dynamic, policy, parent);
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