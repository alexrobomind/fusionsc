#pragma once

#include "fscpy.h"

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/schema.h>

namespace fscpy {

// Converts a scalar (python or numpy) into an instance of DynamicValue	
Maybe<capnp::DynamicValue::Reader> dynamicValueFromScalar(py::handle handle);
	
namespace internal {

template<typename T, typename SFINAE = void>
struct IsBuilder_ {
	static constexpr bool val = false;
};

template<typename T>
struct IsBuilder_<T, kj::VoidSfinae<capnp::FromBuilder<T>>> {
	static constexpr bool val = true;
};

}

template<typename T>
constexpr bool isBuilder() { return internal::IsBuilder_<T>::val; }

namespace internal {
	
template<typename T, typename SFINAE = void>
struct ToReader_ {
	static auto exec(T t) { return t; }
};

template<typename T>
struct ToReader_<T, kj::EnableIf<isBuilder<T>()>> {
	static auto exec(T t) { return t.asReader(); }
};

}

template<typename T>
auto toReader(T t) {
	return internal::ToReader_<T>::exec(t);
}
	
struct DynamicValuePipeline;

namespace internal {
	template<typename T>
	struct GetPipelineAsImpl {
		static_assert(sizeof(T) == 0, "Unsupported type for Pipeline::getAs");
	};
}

//! Conversion helper to obtain DynamicValue from Python and NumPy scalar types
Maybe<capnp::DynamicValue::Reader> dynamicValueFromScalar(py::handle handle);

struct MessageHook : public kj::Refcounted {	
	inline MessageHook(Own<void> data) : data(mv(data)) {}
	Own<MessageHook> addRef() { return kj::addRef(*this); }
	
private:
	Own<void> data;
};

struct WithMessageBase {
	WithMessageBase() = default;
	WithMessageBase(Own<void> target) : hook(kj::refcounted<MessageHook>(mv(target))) {}
	
	inline WithMessageBase(WithMessageBase& other) : hook(other.hook -> addRef()) {}
	WithMessageBase(WithMessageBase&& other) = default;
	
private:
	Own<MessageHook> hook;
};

template<typename T>
struct WithMessage : public T, public WithMessageBase {
	template<typename... Params>
	WithMessage(WithMessageBase msg, Params&&... params) :
		T(fwd<Params>(params)...),
		WithMessageBase(mv(msg))
	{}
	
	template<typename... Params>
	WithMessage(Own<void> msg, Params&&... params) :
		T(fwd<Params>(params)...),
		WithMessageBase(mv(msg))
	{}
	
	// Type-dispatching forwarding constructor
	template<typename T2, typename SFINAE = kj::EnableIf<std::is_base_of<WithMessageBase, T2>::value>>
	WithMessage(T2&& t2) :
		T(fwd<T2>(t2)), WithMessageBase(fwd<T2>(t2))
	{}
	
	T& wrapped() { return *this; }
};

struct DynamicValueReader;
struct DynamicValueBuilder;
struct DynamicValuePipeline;

struct DynamicStructReader;
struct DynamicStructBuilder;
struct DynamicStructPipeline;

struct DynamicListReader;
struct DynamicListBuilder;

struct DataReader;
struct DataBuilder;
struct TextReader;
using TextBuilder = TextReader; // Who the f'ck modifies a text in-place anyway?

struct AnyReader;
struct AnyBuilder;

using DynamicCapabilityClient = capnp::DynamicCapability::Client;
using DynamicCapabilityServer = capnp::DynamicCapability::Server;

struct EnumInterface;

struct DynamicOrphan;

struct FieldDescriptor;

// class Definitions

struct DynamicValueReader : public WithMessage<capnp::DynamicValue::Reader> {
	using WithMessage::WithMessage;
	
	DynamicStructReader asStruct();
	DynamicListReader asList();
	DataReader asData();
	TextReader asText();
	AnyReader asAny();
};

struct DynamicValueBuilder : public WithMessage<capnp::DynamicValue::Builder> {
	using WithMessage::WithMessage;
	
	DynamicStructBuilder asStruct();
	DynamicListBuilder asList();
	DataBuilder asData();
	TextBuilder asText();
	AnyBuilder asAny();
};

namespace internal {

template<typename T, bool ib = isBuilder<T>()>
struct DynamicValueType_ {
	using Type = DynamicValueBuilder;
};

template<typename T>
struct DynamicValueType_<T, false> {
	using Type = DynamicValueReader;
};

}

template<typename T>
using DynamicValueType = typename internal::DynamicValueType_<T>::Type;

struct DynamicValuePipeline {
	capnp::AnyPointer::Pipeline typeless;
	capnp::Schema schema;
	
	inline DynamicValuePipeline(capnp::AnyPointer::Pipeline typeless, capnp::Schema schema) :
		typeless(mv(typeless)),
		schema(mv(schema))
	{}
	
	inline DynamicValuePipeline() : typeless(nullptr), schema() {}
	
	inline DynamicValuePipeline(DynamicValuePipeline& other) :
		typeless(other.typeless.noop()),
		schema(other.schema)
	{}
	
	inline DynamicValuePipeline(DynamicValuePipeline&& other) = default;
	inline DynamicValuePipeline& operator=(DynamicValuePipeline&& other) = default;
	
	DynamicStructPipeline asStruct();
	capnp::DynamicCapability::Client asCapability();
	
	template<typename T>
	auto getAs() {
		return internal::GetPipelineAsImpl<T>::apply(*this);
	}
};

template<typename StructType>
struct DynamicStructInterface : public WithMessage<StructType> {
	using WithMessage<StructType>::WithMessage;
	
	DynamicValueType<StructType> get(kj::StringPtr field);
	DynamicValueType<StructType> getCapnp(capnp::StructSchema::Field field);
	bool has(kj::StringPtr field);
	kj::StringPtr whichStr();
	
	kj::String repr();
	kj::String toYaml(bool flow);
	
	uint32_t size();
	
	py::dict asDict();
	
	py::buffer_info buffer();
	
	DynamicStructBuilder clone();
};

struct DynamicStructReader : public DynamicStructInterface<capnp::DynamicStruct::Reader> {
	using DynamicStructInterface::DynamicStructInterface;
};

struct DynamicStructBuilder : public DynamicStructInterface<capnp::DynamicStruct::Builder> {
	using DynamicStructInterface::DynamicStructInterface;
	
	void set(kj::StringPtr field, py::object value);
	void initList(kj::StringPtr field, uint32_t size);
	DynamicOrphan disown(kj::StringPtr field);
};

struct DynamicStructPipeline {
	capnp::AnyPointer::Pipeline typeless;
	capnp::StructSchema schema;
	
	inline DynamicStructPipeline(capnp::AnyPointer::Pipeline typeless, capnp::StructSchema schema) :
		typeless(mv(typeless)),
		schema(mv(schema))
	{}
	
	inline DynamicStructPipeline() : typeless(nullptr), schema() {}
	
	inline DynamicStructPipeline(DynamicStructPipeline& other) :
		typeless(other.typeless.noop()),
		schema(other.schema)
	{}
	
	inline DynamicStructPipeline(DynamicStructPipeline&& other) = default;
	
	inline capnp::StructSchema getSchema() { return schema; }
	
	DynamicValuePipeline get(kj::StringPtr fieldName);
	DynamicValuePipeline getCapnp(capnp::StructSchema::Field field);
	
	inline kj::StringPtr whichStr() { return ""; }
	inline kj::String repr() { return kj::str("<Pipeline for ", schema, ">"); }
};

template<typename ListType>
struct DynamicListInterface : public WithMessage<ListType> {
	using WithMessage<ListType>::WithMessage;
	
	struct Iterator {
		inline DynamicValueType<ListType> operator*() { return DynamicValueType<ListType>(parent, parent.get(pos)); }
		
		inline bool operator!=(const Iterator& other) const { return pos != other.pos; }
		inline bool operator==(const Iterator& other) const { return pos == other.pos; }
		inline Iterator& operator++() { ++pos; return *this; }
		
		inline Iterator(DynamicListInterface& parent, size_t idx) : parent(parent), pos(pos) {}
	private:
		DynamicListInterface& parent;
		uint32_t pos;
	};
	
	inline Iterator begin() { return Iterator(*this, 0); }
	inline Iterator end() { return Iterator(*this, ListType::size()); }
	
	DynamicValueType<ListType> get(uint32_t idx);
		
	py::buffer_info buffer();
	
	DynamicListBuilder clone();
};

struct DynamicListReader : public DynamicListInterface<capnp::DynamicList::Reader> {
	using DynamicListInterface::DynamicListInterface;
};

struct DynamicListBuilder : public DynamicListInterface<capnp::DynamicList::Builder> {	
	using DynamicListInterface::DynamicListInterface;
	// Builder interface
	void set(uint32_t, py::object value);
	void initList(uint32_t idx, uint32_t size);
};

struct DataReader : public WithMessage<capnp::Data::Reader> {
	using WithMessage::WithMessage;
	
	py::buffer_info buffer();
	kj::String repr();
};

struct DataBuilder : public WithMessage<capnp::Data::Builder> {
	using WithMessage::WithMessage;
	
	py::buffer_info buffer();
	kj::String repr();
};

struct TextReader : public WithMessage<capnp::Text::Reader> {
	using WithMessage::WithMessage;
	
	kj::StringPtr repr();
};

struct AnyReader : public WithMessage<capnp::AnyPointer::Reader> {
	using WithMessage::WithMessage;
	
	kj::String repr();
};

struct AnyBuilder : public WithMessage<capnp::AnyPointer::Builder> {
	using WithMessage::WithMessage;
	
	kj::String repr();
	
	void setList(DynamicListReader);
	void setStruct(DynamicStructReader);
	void adopt(DynamicOrphan&);
	
	DynamicOrphan disown();
};

struct DynamicOrphan : private WithMessage<capnp::Orphan<capnp::DynamicValue>> {
	using WithMessage::WithMessage;
	
	bool consumed = false;
	
	DynamicValueBuilder get();
	capnp::Orphan<capnp::DynamicValue>&& release();
};

struct EnumInterface : public capnp::DynamicEnum {
	using DynamicEnum::DynamicEnum;
	
	kj::String repr();
	bool eq1(DynamicEnum& other);
	bool eq2(uint16_t other);
	bool eq3(kj::StringPtr other);
};

//! Implements the descriptor protocol for Cap'n'Proto fields
struct FieldDescriptor : public capnp::StructSchema::Field {
	using Field::Field;
	
	DynamicValueBuilder get1(DynamicStructBuilder);
	DynamicValueReader get2(DynamicStructReader);
	kj::String get3(py::none, py::type);
	
	void set(DynamicStructBuilder obj, py::object value);
	void del(DynamicStructBuilder obj);
};

}