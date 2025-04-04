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
	
	#define FSCPY_DEF_DISPATCH(T, func) \
		template<> \
		struct PolymorphicDispatchTraits<T> { \
			static py::type get(T& val) { return defaultLoader.func(val.getSchema()); } \
		};\
	
	FSCPY_DEF_DISPATCH(DynamicCapabilityClient, clientType);
	FSCPY_DEF_DISPATCH(DynamicStructBuilder, builderType);
	FSCPY_DEF_DISPATCH(DynamicStructReader, readerType);
	FSCPY_DEF_DISPATCH(DynamicStructPipeline, pipelineType);
	
	inline void checkSame(capnp::Schema expected, capnp::Schema provided) {
		if(provided == expected)
			return;
		
		KJ_REQUIRE(provided.getProto().getId() == expected.getProto().getId(), "Schema type mismatch", expected, provided);
		
		Temporary<capnp::schema::Brand> providedBrand;
		Temporary<capnp::schema::Brand> expectedBrand;
		
		fsc::extractBrand(provided, providedBrand);
		fsc::extractBrand(expected, expectedBrand);
		
		{
			auto schema = expected;
			auto provided = providedBrand.asReader();
			auto expected = expectedBrand.asReader();
			
			KJ_FAIL_REQUIRE("Schema brand mismatch for schema", schema, expected, provided);
		}
	}
};

namespace pybind11 { namespace detail {
	
	// Polymorphic casters for dynamic classes
		
	template<typename T>
	struct PolymorphicDispatchCaster : public type_caster_base<T> {	
		PolymorphicDispatchCaster() = default;
		PolymorphicDispatchCaster(const PolymorphicDispatchCaster&) = delete;
		PolymorphicDispatchCaster(PolymorphicDispatchCaster&&) = default;
		
		static handle cast(T src, return_value_policy policy, handle parent) {
			object baseInstance = reinterpret_steal<object>(type_caster_base<T>::cast(kj::mv(src), policy, parent));
			type derivedClass = ::fscpy::PolymorphicDispatchTraits<T>::get(src);
			return derivedClass(baseInstance).inc_ref();
		}
	};
	
	template<>
	struct type_caster<fscpy::DynamicCapabilityClient> : public PolymorphicDispatchCaster<fscpy::DynamicCapabilityClient>
	{
		using Base = PolymorphicDispatchCaster<fscpy::DynamicCapabilityClient>;
		
		inline operator fscpy::DynamicCapabilityClient&() {
			auto& baseResult = Base::operator fscpy::DynamicCapabilityClient&();
			KJ_REQUIRE(baseResult.locallyOwned());
			return baseResult;
		}
	};
	
	template<>
	struct type_caster<fscpy::DynamicStructBuilder> : public PolymorphicDispatchCaster<fscpy::DynamicStructBuilder>
	{};
	
	template<>
	struct type_caster<fscpy::DynamicStructReader> : public PolymorphicDispatchCaster<fscpy::DynamicStructReader>
	{};
	
	template<>
	struct type_caster<fscpy::DynamicStructPipeline> : public PolymorphicDispatchCaster<fscpy::DynamicStructPipeline> {	
	};
	
	template<>
	struct type_caster<capnp::DynamicCapability::Client> : public type_caster<fscpy::DynamicCapabilityClient> {
		inline operator capnp::DynamicCapability::Client&() {
			return operator fscpy::DynamicCapabilityClient&();
		}
		
		/*inline operator capnp::DynamicCapability::Client&&() {
			return operator fscpy::DynamicCapabilityClient&&();
		}*/
	};
	
	// classes for capnp::Type objects
	
	template<>
	struct type_caster<capnp::Type> /*: public type_caster_base<capnp::Type>*/ {	
		
		PYBIND11_TYPE_CASTER(capnp::Type, const_name("fusionsc.capnp.Type"));	
		static handle cast(capnp::Type type, return_value_policy policy, handle parent) {
			if(type.isStruct()) {
				return type_caster<capnp::StructSchema>::cast(type.asStruct(), return_value_policy::move, parent);
			}
			
			if(type.isInterface()) {
				return type_caster<capnp::InterfaceSchema>::cast(type.asInterface(), return_value_policy::move, parent);
			}
			
			if(type.isEnum()) {
				return type_caster<capnp::EnumSchema>::cast(type.asEnum(), return_value_policy::move, parent);
			}
			
			if(type.isList()) {
				return type_caster<capnp::ListSchema>::cast(type.asList(), return_value_policy::move, parent);
			}
			
			return type_caster_base<capnp::Type>::cast(type, return_value_policy::move, parent);
		}
		
		bool load(py::handle src, bool convert) {
			#define FSCPY_TRY_CAST(T) {\
				type_caster<T> caster; \
				if(caster.load(src, convert)) { \
					value = caster.operator T&(); \
					return true; \
				} \
			}
			FSCPY_TRY_CAST(capnp::StructSchema);
			FSCPY_TRY_CAST(capnp::InterfaceSchema);
			FSCPY_TRY_CAST(capnp::EnumSchema);
			FSCPY_TRY_CAST(capnp::ListSchema);
			
			#undef FSCPY_TRY_CAST
			
			type_caster_base<capnp::Type> baseCaster;
			if(baseCaster.load(src, convert)) {
				value = baseCaster.operator capnp::Type&();
				return true;
			}
			
			return false;
		}
	};
	
	template<>
	struct type_caster<capnp::Schema> : public type_caster_base<capnp::Schema> {		
		static handle cast(capnp::Schema schema, return_value_policy policy, handle parent) {
			using Node = capnp::schema::Node;
			
			auto proto = schema.getProto();
			switch(proto.which()) {
				case Node::STRUCT:
					return type_caster<capnp::StructSchema>::cast(schema.asStruct(), return_value_policy::move, parent);
				case Node::ENUM:
					return type_caster<capnp::EnumSchema>::cast(schema.asEnum(), return_value_policy::move, parent);
				case Node::INTERFACE:
					return type_caster<capnp::InterfaceSchema>::cast(schema.asInterface(), return_value_policy::move, parent);
				case Node::CONST:
					return type_caster<capnp::ConstSchema>::cast(schema.asConst(), return_value_policy::move, parent);
				default:
					break;
			}
			
			return type_caster_base<capnp::Schema>::cast(schema, return_value_policy::move, parent);
		}
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
				capnp::StructSchema staticSchema = fscpy::defaultLoader.schemaFor<Builds>().asStruct();
				
				//KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				::fscpy::checkSame(staticSchema, dynamic.getSchema());
				ASB any = dynamic.wrapped();
			
				value = fscpy::WithMessage<Builder>(fscpy::shareMessage(dynamic), any.as<Builds>());
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(fscpy::WithMessage<Builder> src, return_value_policy policy, handle parent) {
			capnp::StructSchema schema = fscpy::defaultLoader.schemaFor<Builds>().asStruct();
			
			ASB any = capnp::toAny(src.wrapped());
			DSB dynamic(fscpy::shareMessage(src), any.as<capnp::DynamicStruct>(schema));
			
			return type_caster<DSB>::cast(dynamic, return_value_policy::move, parent);
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
			using BuilderType = fscpy::WithMessage<typename capnp::FromReader<Reader>::Builder>;
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
				capnp::StructSchema staticSchema = fscpy::defaultLoader.schemaFor<Reads>().asStruct();
				
				// KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				::fscpy::checkSame(staticSchema, dynamic.getSchema());
				ASR any = dynamic.wrapped();
			
				value = fscpy::WithMessage<Reader>(fscpy::shareMessage(dynamic), any.as<Reads>());
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(fscpy::WithMessage<Reader> src, return_value_policy policy, handle parent) {
			capnp::StructSchema schema = fscpy::defaultLoader.schemaFor<Reads>().asStruct();
			
			ASR any = capnp::toAny(src.wrapped());
			DSR dynamic(fscpy::shareMessage(src), any.as<capnp::DynamicStruct>(schema));
			
			return type_caster<DSR>::cast(dynamic, return_value_policy::move, parent);
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
		
		template<>
		struct CastThisCap_<fscpy::DynamicCapabilityClient> { static constexpr bool val = false; };
		
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
			type_caster<fscpy::DynamicCapabilityClient> subCaster;
			if(!subCaster.load(src, convert)) {
				return false;
			}
			
			try {
				capnp::DynamicCapability::Client dynamic = (capnp::DynamicCapability::Client&) subCaster;
				capnp::InterfaceSchema staticSchema = fscpy::defaultLoader.schemaFor<ClientFor>().asInterface();
				
				// KJ_REQUIRE(dynamic.getSchema() == staticSchema, "Incompatible types");
				::fscpy::checkSame(staticSchema, dynamic.getSchema());
				capnp::Capability::Client any = dynamic;
			
				value = dynamic.castAs<ClientFor>();
			} catch(kj::Exception e) {
				KJ_LOG(WARNING, "Error during conversion", e);
				return false;
			}
			
			return true;
		}
		
		static handle cast(Client src, return_value_policy policy, handle parent) {
			capnp::InterfaceSchema schema = fscpy::defaultLoader.schemaFor<ClientFor>().asInterface();
			
			capnp::Capability::Client anyCap = src;
			capnp::DynamicCapability::Client dynamic = src.template castAs<capnp::DynamicCapability>(schema);
			
			return type_caster<fscpy::DynamicCapabilityClient>::cast(dynamic, return_value_policy::move, parent);
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
				case DV::UNKNOWN: KJ_FAIL_REQUIRE("Unknown builder type");
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
				value = fscpy::TextReader::from(kj::heapString(py::cast<kj::StringPtr>(src)));
				return true;
			}
			
			if(py::isinstance<py::bytes>(src) || py::isinstance<py::memoryview>(src) || py::isinstance<py::bytearray>(src)) {
				Py_buffer view;
				
				if(PyObject_GetBuffer(src.ptr(), &view, PyBUF_SIMPLE) != 0)
					throw py::error_already_set();
				
				KJ_DEFER({ PyBuffer_Release(&view); });
				
				value = fscpy::DataReader(
					kj::heap(py::reinterpret_borrow<py::object>(src)),
					kj::ArrayPtr<const kj::byte>(
						(const kj::byte*) view.buf,
						(size_t) view.len
					)
				);
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
				case DV::UNKNOWN: KJ_FAIL_REQUIRE("Unknown builder type");
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
