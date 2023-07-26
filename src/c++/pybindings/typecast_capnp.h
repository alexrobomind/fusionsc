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
//   fscpy::DynamicValueBuilder
//   fscpy::DynamicValueReader

//   fscpy::DynamicStructBuilder
//   fscpy::DynamicStructReader
//   fscpy::DynamicStructPipeline
//   fscpy::DynamicCapabilityClient (a.k.a. capnp::DynamicCapability::Client)
//
//   capnp struct readers
//   capnp struct builders
//   capnp capabilities
//
//   fsc::Temporary<...>
//
//   fscpy::DynamicStructPipeline

namespace fscpy {
	template<typename T>
	struct PolymorphicDispatchTraits {
		static_assert(sizeof(T) == 0, "No polymorphic dispatch specified for this class");
	};
	
	#define FSCPY_DEF_DISPATCH(T, attr) \
		template<> \
		struct PolymorphicDispatchTraits<T> { \
			static const inline kj::StringPtr attributeName = attr; \
		};\
	
	FSCPY_DEF_DISPATCH(DynamicCapabilityClient, "Client");
	FSCPY_DEF_DISPATCH(DynamicStructBuilder, "Builder");
	FSCPY_DEF_DISPATCH(DynamicStructReader, "Reader");
	FSCPY_DEF_DISPATCH(DynamicStructPipeline, "Pipeline");		
};

namespace pybind11 { namespace detail {
	
	// Polymorphic casters for dynamic classes
		
	template<typename T>
	struct PolymorphicDispatchCaster : public type_caster_base<T> {	
		PolymorphicDispatchCaster() = default;
		PolymorphicDispatchCaster(const PolymorphicDispatchCaster&) = delete;
		PolymorphicDispatchCaster(PolymorphicDispatchCaster&&) = default;
		
		static handle cast(T src, return_value_policy policy, handle parent) {
			auto typeId = src.getSchema().getProto().getId();
			
			object baseInstance = reinterpret_steal<object>(type_caster_base<T>::cast(kj::mv(src), policy, parent));
			
			// Look for a derived class
			if(globalClasses->contains(typeId)) {
				auto derivedClass = (*globalClasses)[py::cast(typeId)].attr(py::cast(fscpy::PolymorphicDispatchTraits<T>::attributeName));
				
				object derivedInstance = derivedClass(baseInstance);
				return derivedInstance.inc_ref();
			}
			
			return baseInstance.inc_ref();
		}
	};
	
	template<>
	struct type_caster<fscpy::DynamicCapabilityClient> : public PolymorphicDispatchCaster<fscpy::DynamicCapabilityClient>
	{};
	
	template<>
	struct type_caster<fscpy::DynamicStructBuilder> : public PolymorphicDispatchCaster<fscpy::DynamicStructBuilder>
	{};
	
	template<>
	struct type_caster<fscpy::DynamicStructReader> : public PolymorphicDispatchCaster<fscpy::DynamicStructReader>
	{};
	
	template<>
	struct type_caster<fscpy::DynamicStructPipeline> : public PolymorphicDispatchCaster<fscpy::DynamicStructPipeline> {	
	};
	
	// Dynamic <-> Static casters for static classes w/ messages
			
	template<typename Builder>
	struct type_caster<fscpy::WithMessage<Builder>, kj::EnableIf<CAPNP_KIND(capnp::FromBuilder<Builder>) == capnp::Kind::STRUCT>> {
		using DSB = fscpy::DynamicStructBuilder;
		using ASB  = capnp::AnyStruct::Builder;
		using Builds = typename Builder::Builds;
		
		PYBIND11_TYPE_CASTER(fscpy::WithMessage<Builder>, const_name<Builds>() + const_name(".Builder"));
		
		type_caster() : value(nullptr, nullptr) {}
		type_caster(const type_caster&) = delete;
		type_caster(type_caster&&) = default;
		
		bool load(handle src, bool convert) {
			// Try to load as dynamic struct
			type_caster<DSB> subCaster;
			if(!subCaster.load(src, convert))
				return false;
			
			try {
				DSB dynamic = static_cast<DSB&>(subCaster);
				capnp::StructSchema staticSchema = fscpy::defaultLoader.importBuiltin<Builds>().asStruct();
				
				KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				ASB any = dynamic.wrapped();
			
				value = fscpy::WithMessage<Builder>(fscpy::shareMessage(dynamic), any.as<Builds>());
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(fscpy::WithMessage<Builder> src, return_value_policy policy, handle parent) {
			capnp::StructSchema schema = fscpy::defaultLoader.importBuiltin<Builds>().asStruct();
			
			ASB any = capnp::toAny(src.wrapped());
			DSB dynamic(fscpy::shareMessage(src), any.as<capnp::DynamicStruct>(schema));
			
			return type_caster<DSB>::cast(dynamic, policy, parent);
		}		
	};
	
	template<typename Reader>
	struct type_caster<fscpy::WithMessage<Reader>, kj::EnableIf<CAPNP_KIND(capnp::FromReader<Reader>) == capnp::Kind::STRUCT>> {
		using DSB = fscpy::DynamicStructBuilder;
		using DSR = fscpy::DynamicStructReader;
		using ASR  = capnp::AnyStruct::Reader;
		using Reads = typename Reader::Reads;
		
		PYBIND11_TYPE_CASTER(fscpy::WithMessage<Reader>, const_name<Reads>() + const_name(".Reader"));

		type_caster() : value(nullptr) {}
		type_caster(const type_caster&) = delete;
		type_caster(type_caster&&) = default;
		
		bool load(handle src, bool convert) {	
			// Builder caster
			using BuilderType = typename fscpy::WithMessage<capnp::FromReader<Reader>::Builder>;
			type_caster<BuilderType> builderCaster;
			if(builderCaster.load(src, convert)) {
				auto& builder = (BuilderType&) builderCaster;
				
				value = fscpy::WithMessage<Reader>(fscpy::shareMessage(builder), builder.asReader());
				return true;
			}
			
			// Try to load as dynamic struct
			type_caster<DSR> subCaster;
			if(!subCaster.load(src, convert))
				return false;
			
			try {
				DSR dynamic = static_cast<DSR&>(subCaster);
				capnp::StructSchema staticSchema = fscpy::defaultLoader.importBuiltin<Reads>().asStruct();
				
				KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				ASR any = dynamic.wrapped();
			
				value = fscpy::WithMessage<Reader>(fscpy::shareMessage(dynamic), any.as<Reads>());
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(fscpy::WithMessage<Reader> src, return_value_policy policy, handle parent) {
			capnp::StructSchema schema = fscpy::defaultLoader.importBuiltin<Reads>().asStruct();
			
			ASR any = capnp::toAny(src.wrapped());
			DSR dynamic(fscpy::shareMessage(src), any.as<capnp::DynamicStruct>(schema));
			
			return type_caster<DSR>::cast(dynamic, policy, parent);
		}		
	};
	
	template<typename T>
	struct type_caster<fsc::Temporary<T>, kj::EnableIf<capnp::kind<T>() == capnp::Kind::STRUCT>> {
		PYBIND11_TYPE_CASTER(fsc::Temporary<T>, const_name<T>() + const_name(".Builder"));
		FSCPY_MOVE_ONLY_CASTER;
		
		bool load(handle src, bool convert) {
			return false;
		}
		
		static handle cast(fsc::Temporary<T> src, return_value_policy policy, handle parent) {
			using T2 = fscpy::WithMessage<typename T::Builder>;
			
			return type_caster<T2>::cast(
				T2(mv(src.holder), src.asBuilder()), 
				policy, parent
			);
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
		
		type_caster() : value(nullptr) {}
		type_caster(const type_caster&) = delete;
		type_caster(type_caster&&) = default;
		
		bool load(handle src, bool convert) {
			// Try to load as dynamic struct
			type_caster<capnp::DynamicCapability::Client> subCaster;
			if(!subCaster.load(src, convert)) {
				return false;
			}
			
			try {
				capnp::DynamicCapability::Client dynamic = (capnp::DynamicCapability::Client&) subCaster;
				capnp::InterfaceSchema staticSchema = fscpy::defaultLoader.importBuiltin<ClientFor>().asInterface();
				
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
	
	// Dynamic value casters
	
	template<>
	struct type_caster<fscpy::DynamicValueBuilder> {		
		PYBIND11_TYPE_CASTER(fscpy::DynamicValueBuilder, const_name("DynamicValueBuilder"));
		FSCPY_MOVE_ONLY_CASTER;
				
		bool load(handle src, bool convert) {			
			#define FSCPY_TRY_CAST(Type) \
				{ \
					type_caster<Type> caster; \
					if(caster.load(src, convert)) { \
						value = (Type&) caster; \
						return true; \
					} \
				}		
			
			FSCPY_TRY_CAST(fscpy::DynamicStructBuilder)
			FSCPY_TRY_CAST(fscpy::DynamicListBuilder)
			FSCPY_TRY_CAST(fscpy::AnyBuilder)
			FSCPY_TRY_CAST(fscpy::EnumInterface)
			FSCPY_TRY_CAST(fscpy::DynamicCapabilityClient)
			
			#undef FSCPY_TRY_CAST
			return false;
		}
		
		static handle cast(fscpy::DynamicValueBuilder src, return_value_policy policy, handle parent) {
			using DV = capnp::DynamicValue;
			
			switch(src.getType()) {
				case DV::VOID: return none().inc_ref();
				case DV::BOOL: return py::cast(src.as<bool>()).inc_ref();
				case DV::INT: return py::cast(src.as<int64_t>()).inc_ref();
				case DV::UINT: return py::cast(src.as<uint64_t>()).inc_ref();
				case DV::FLOAT: return py::cast(src.as<double>()).inc_ref();
				case DV::TEXT: return py::cast(src.asText()).inc_ref();
				case DV::DATA: return py::cast(src.asData()).inc_ref();
				case DV::LIST: return py::cast(src.asList()).inc_ref();
				case DV::STRUCT: return py::cast(src.asStruct()).inc_ref();
				case DV::ENUM: return py::cast(src.asEnum()).inc_ref();
				case DV::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>()).inc_ref();
				case DV::ANY_POINTER: return py::cast(src.asAny()).inc_ref();
			}
			
			KJ_UNREACHABLE;
		}
	};
	
	template<>
	struct type_caster<fscpy::DynamicValueReader> {		
		PYBIND11_TYPE_CASTER(fscpy::DynamicValueReader, const_name("DynamicValueReader"));
		FSCPY_MOVE_ONLY_CASTER;	
		
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
				value = fscpy::TextReader::from(py::cast<kj::String>(src));
				return true;
			}
			
			if(py::isinstance<py::bytes>(src)) {
				char *buffer = nullptr;
				ssize_t length = 0;
				auto asBytes = py::reinterpret_borrow<py::bytes>(src);
				
				if(PyBytes_AsStringAndSize(asBytes.ptr(), &buffer, &length) != 0)
					throw py::error_already_set();
				
				value = fscpy::DataReader(kj::heap(src), kj::ArrayPtr<const kj::byte>((const kj::byte*) buffer, (size_t) length));
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
			
	
			FSCPY_TRY_CAST(fscpy::DynamicValueBuilder)
			FSCPY_TRY_CAST(fscpy::DynamicStructReader)
			FSCPY_TRY_CAST(fscpy::DynamicListReader)
			FSCPY_TRY_CAST(fscpy::AnyReader)
			FSCPY_TRY_CAST(fscpy::EnumInterface)
			FSCPY_TRY_CAST(fscpy::DynamicCapabilityClient)
			
			#undef FSCPY_TRY_CAST
			
			return false;
		}
		
		
		static handle cast(fscpy::DynamicValueReader src, return_value_policy policy, handle parent) {
			using DV = capnp::DynamicValue;
			
			switch(src.getType()) {
				case DV::VOID: return none().inc_ref();
				case DV::BOOL: return py::cast(src.as<bool>()).inc_ref();
				case DV::INT: return py::cast(src.as<int64_t>()).inc_ref();
				case DV::UINT: return py::cast(src.as<uint64_t>()).inc_ref();
				case DV::FLOAT: return py::cast(src.as<double>()).inc_ref();
				case DV::TEXT: return py::cast(src.asText()).inc_ref();
				case DV::DATA: return py::cast(src.asData()).inc_ref();
				case DV::LIST: return py::cast(src.asList()).inc_ref();
				case DV::ENUM: return py::cast(src.asEnum()).inc_ref();
				case DV::STRUCT: return py::cast(src.asStruct()).inc_ref();
				case DV::ANY_POINTER: return py::cast(src.asAny()).inc_ref();
				case DV::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>()).inc_ref();
			}
			
			KJ_UNREACHABLE;
		}
	};
	
	template<>
	struct type_caster<fscpy::DynamicValuePipeline> {
		PYBIND11_TYPE_CASTER(fscpy::DynamicValuePipeline, const_name("DynamicValuePipeline"));
		FSCPY_MOVE_ONLY_CASTER;
		
		bool load(handle src, bool convert) {
			/* Currently, the machinery to create DynamicValuePipeline instances
			   from instances of specific pipeline classes does not exist (might
			   require a PipelineBuilder object. */
			
			return false;
			   
			/*#define FSCPY_TRY_CAST(Type) \
				{ \
					type_caster<Type> caster; \
					if(caster.load(src, convert)) { \
						value = (Type&) caster; \
						return true; \
					} \
				}
						
			FSCPY_TRY_CAST(fscpy::DynamicStructPipeline)
			FSCPY_TRY_CAST(fscpy::DynamicCapabilityClient)
			
			#undef FSCPY_TRY_CAST
			*/
		}
		
		
		static handle cast(fscpy::DynamicValuePipeline src, return_value_policy policy, handle parent) {
			auto schema = src.schema;
			
			switch(schema.getProto().which()) {
				case capnp::schema::Node::STRUCT: return py::cast(src.asStruct()).inc_ref();
				case capnp::schema::Node::INTERFACE: return py::cast(src.asCapability()).inc_ref();
				default: KJ_FAIL_REQUIRE("Invalid pipeline type, must be struct or interface", schema.getProto().which());
			}
		}
	};
}}