#pragma once

#include "fscpy.h"

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/schema.h>

namespace fscpy {

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
struct TextBuilder; // Who the f'ck modifies a text in-place anyway?

struct AnyReader;
struct AnyBuilder;

using DynamicCapabilityClient = capnp::DynamicCapability::Client;
using DynamicCapabilityServer = capnp::DynamicCapability::Server;

struct EnumInterface;

struct DynamicOrphan;

struct FieldDescriptor;

/** 
 * Catch-all converter for various things that look like scalar numbers from the python side.
 * 
 * Python has a massive amount of different things that could possibly
 * be counted as "scalar numeric values", including:
 *  - Baseline numeric types (boxed numbers of variable width ints and floats - int, float, bool)
 *  - Numpy array scalars (boxed numbers of fixed-width types)
 *  - 0D numpy arrays (which are NOT the same as scalars, ugh)
 */
Maybe<DynamicValueReader> dynamicValueFromScalar(py::handle handle);
	
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

struct MessageHook : public kj::Refcounted {	
	inline MessageHook(Own<const void> data) : data(mv(data)) {}
	Own<MessageHook> addRef() { return kj::addRef(*this); }
	
private:
	Own<const void> data;
};

struct WithMessageBase {
	WithMessageBase() = default;
	WithMessageBase(Own<const void> target) : hook(kj::refcounted<MessageHook>(mv(target))) {}
	
	inline WithMessageBase(WithMessageBase& other) : hook(other.hook -> addRef()) {}
	WithMessageBase(WithMessageBase&& other) = default;
	
	inline WithMessageBase& operator=(WithMessageBase& other) { hook = other.hook -> addRef(); return *this; }
	inline WithMessageBase& operator=(WithMessageBase&& other) = default;
	
private:
	Own<MessageHook> hook;
};

template<typename T>
struct CopyMsgT {
	kj::Decay<T>& ref;
};

template<typename T>
struct ShareMsgT {
	kj::Decay<T>& ref;
};

template<typename T>
struct NoMsgT {
	kj::Decay<T>& ref;
};

template<typename T>
CopyMsgT<T> copyMessage(T&& t) {
	CopyMsgT<T> result { t };
	return result;
}

template<typename T>
ShareMsgT<T> shareMessage(T&& t) {
	ShareMsgT<T> result { t };
	return result;
}

template<typename T>
NoMsgT<T> noMessage(T&& t) {
	NoMsgT<T> result { t };
	return result;
}

struct noMessage {};

template<typename T>
struct WithMessage : public T, public WithMessageBase {
	static inline const int ANONYMOUS = 0;
	
	template<typename T2, typename... Params>
	WithMessage(ShareMsgT<T2> input, Params&&... params) :
		T(fwd<Params>(params)...),
		WithMessageBase(fwd<T2>(input.ref))
	{}
	
	template<typename T2>
	WithMessage(CopyMsgT<T2> input) :
		T(fwd<T2>(input.ref)),
		WithMessageBase(fwd<T2>(input.ref))
	{}
	
	template<typename... Params>
	WithMessage(decltype(nullptr), Params&&... params) :
		T(fwd<Params>(params)...),
		WithMessageBase(kj::attachRef(ANONYMOUS))
	{}
	
	template<typename T2>
	WithMessage(NoMsgT<T2> input) :
		WithMessage(nullptr, fwd<T2>(input.ref))
	{}
	
	template<typename... Params>
	WithMessage(Own<const void> msg, Params&&... params) :
		T(fwd<Params>(params)...),
		WithMessageBase(mv(msg))
	{}
	
	WithMessage(WithMessage& other) = default;
	WithMessage(WithMessage&& other) = default;
	WithMessage& operator=(WithMessage& other) = default;
	WithMessage& operator=(WithMessage&& other) = default;
	
	T& wrapped() { return *this; }
};

template<typename T, typename MsgSrc>
WithMessage<T> bundleWithMessage(T&& input, MsgSrc&& src) {
	return WithMessage<T>(fwd<MsgSrc>(src), fwd<T>(input));
}

#define COPY_WITH_MSG(Cls, Input) \
	inline Cls (Input& i) : Cls(copyMessage(i)) {}; \
	inline Cls (Input&& i) : Cls(copyMessage(mv(i))) {};

#define COPY_WITHOUT_MSG(Cls, Input) \
	inline Cls (Input& i) : Cls(nullptr, i) {}; \
	inline Cls (Input&& i) : Cls(nullptr, mv(i)) {};

// class Definitions

struct DynamicValueReader : public WithMessage<capnp::DynamicValue::Reader> {
	using WithMessage::WithMessage;
	
	inline DynamicValueReader() : WithMessage(nullptr, capnp::Void()) {};
	
	DynamicValueReader(DynamicValueBuilder&);
	
	COPY_WITH_MSG(DynamicValueReader, DynamicStructReader);
	COPY_WITH_MSG(DynamicValueReader, DynamicListReader);
	COPY_WITH_MSG(DynamicValueReader, DataReader);
	COPY_WITH_MSG(DynamicValueReader, TextReader);
	COPY_WITH_MSG(DynamicValueReader, AnyReader);
	
	COPY_WITHOUT_MSG(DynamicValueReader, capnp::Void);
	COPY_WITHOUT_MSG(DynamicValueReader, capnp::DynamicEnum);
	COPY_WITHOUT_MSG(DynamicValueReader, capnp::DynamicCapability::Client);
	COPY_WITHOUT_MSG(DynamicValueReader, double);
	COPY_WITHOUT_MSG(DynamicValueReader, int64_t);
	COPY_WITHOUT_MSG(DynamicValueReader, uint64_t);
	COPY_WITHOUT_MSG(DynamicValueReader, bool);
	
	DynamicStructReader asStruct();
	DynamicListReader asList();
	DataReader asData();
	TextReader asText();
	AnyReader asAny();
	EnumInterface asEnum();
	
	DynamicValueBuilder clone();
};

struct DynamicValueBuilder : public WithMessage<capnp::DynamicValue::Builder> {
	using WithMessage::WithMessage;
	
	inline DynamicValueBuilder() : WithMessage(nullptr, capnp::Void()) {};
	
	COPY_WITH_MSG(DynamicValueBuilder, DynamicStructBuilder);
	COPY_WITH_MSG(DynamicValueBuilder, DynamicListBuilder);
	COPY_WITH_MSG(DynamicValueBuilder, DataBuilder);
	COPY_WITH_MSG(DynamicValueBuilder, TextBuilder);
	COPY_WITH_MSG(DynamicValueBuilder, AnyBuilder);
	
	COPY_WITHOUT_MSG(DynamicValueBuilder, capnp::Void);
	COPY_WITHOUT_MSG(DynamicValueBuilder, capnp::DynamicEnum);
	COPY_WITHOUT_MSG(DynamicValueBuilder, capnp::DynamicCapability::Client);
	COPY_WITHOUT_MSG(DynamicValueBuilder, double);
	COPY_WITHOUT_MSG(DynamicValueBuilder, int64_t);
	COPY_WITHOUT_MSG(DynamicValueBuilder, uint64_t);
	COPY_WITHOUT_MSG(DynamicValueBuilder, bool);
	
	DynamicStructBuilder asStruct();
	DynamicListBuilder asList();
	DataBuilder asData();
	TextBuilder asText();
	AnyBuilder asAny();
	EnumInterface asEnum();
	
	DynamicValueBuilder clone();
	
	static DynamicValueBuilder cloneFrom(capnp::DynamicValue::Reader reader);
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
	
	DynamicStructReader encodeSchema();
	
	uint32_t size();
	uint64_t totalBytes();
	
	py::dict asDict();
	
	py::buffer_info buffer();
	
	DynamicStructBuilder clone();
	
	struct Iterator {
		inline kj::StringPtr operator*() {
			auto nonUnion = parent.getSchema().getNonUnionFields();
			
			if(pos < nonUnion.size())
				return nonUnion[pos].getProto().getName();
			
			return parent.whichStr();
		}
		
		inline bool operator!=(const Iterator& other) const { return pos != other.pos; }
		inline bool operator==(const Iterator& other) const { return pos == other.pos; }
		inline Iterator& operator++() { ++pos; return *this; }
		
		inline Iterator(DynamicStructInterface& parent, size_t pos) : parent(parent), pos(pos) {}
		
	private:
		DynamicStructInterface& parent;
		uint32_t pos;
	};
	
	inline Iterator begin() { return Iterator(*this, 0); }
	inline Iterator end() { return Iterator(*this, this -> size()); }
};

struct DynamicStructReader : public DynamicStructInterface<capnp::DynamicStruct::Reader> {
	using DynamicStructInterface::DynamicStructInterface;
	
	DynamicStructReader(DynamicStructBuilder other);
};

struct DynamicStructBuilder : public DynamicStructInterface<capnp::DynamicStruct::Builder> {
	using DynamicStructInterface::DynamicStructInterface;
	
	void set(kj::StringPtr field, py::object value);
	DynamicValueBuilder init(kj::StringPtr field);
	DynamicListBuilder initList(kj::StringPtr field, uint32_t size);
	DynamicOrphan disown(kj::StringPtr field);
	
	static DynamicStructBuilder cloneFrom(capnp::DynamicStruct::Reader reader);
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
	
	inline DynamicStructPipeline& operator=(DynamicStructPipeline& other)
	{
		typeless = other.typeless.noop();
		schema = other.schema;
		return *this;
	}
	
	inline DynamicStructPipeline(DynamicStructPipeline&& other) = default;
	
	inline capnp::StructSchema getSchema() { return schema; }
	
	DynamicValuePipeline get(kj::StringPtr fieldName);
	DynamicValuePipeline getCapnp(capnp::StructSchema::Field field);
	
	inline DynamicStructPipeline clone() { return *this; }
	
	inline kj::StringPtr whichStr() { return ""; }
	inline kj::String repr() { return kj::str("<Pipeline for ", schema, ">"); }
	
	inline uint32_t size() { return schema.getNonUnionFields().size(); }
	
	struct Iterator {
		inline kj::StringPtr operator*() { return parent.schema.getNonUnionFields()[pos].getProto().getName(); }
		
		inline bool operator!=(const Iterator& other) const { return pos != other.pos; }
		inline bool operator==(const Iterator& other) const { return pos == other.pos; }
		inline Iterator& operator++() { ++pos; return *this; }
		
		inline Iterator(DynamicStructPipeline& parent, size_t pos) : parent(parent), pos(pos) {}
		
	private:
		DynamicStructPipeline& parent;
		uint32_t pos;
	};
	
	inline Iterator begin() { return Iterator(*this, 0); }
	inline Iterator end() { return Iterator(*this, this -> size()); }
};

template<typename ListType>
struct DynamicListInterface : public WithMessage<ListType> {
	using WithMessage<ListType>::WithMessage;
	
	DynamicValueType<ListType> get(uint32_t idx);
	py::buffer_info buffer();
	DynamicListBuilder clone();
	
	kj::String repr();
	kj::String toYaml(bool flow);
	
	struct Iterator {
		inline DynamicValueType<ListType> operator*() { return DynamicValueType<ListType>(shareMessage(parent), parent.get(pos)); }
		
		inline bool operator!=(const Iterator& other) const { return pos != other.pos; }
		inline bool operator==(const Iterator& other) const { return pos == other.pos; }
		inline Iterator& operator++() { ++pos; return *this; }
		
		inline Iterator(DynamicListInterface& parent, size_t pos) : parent(parent), pos(pos) {}
		
	private:
		DynamicListInterface& parent;
		uint32_t pos;
	};
	
	inline Iterator begin() { return Iterator(*this, 0); }
	inline Iterator end() { return Iterator(*this, ListType::size()); }
};

struct DynamicListReader : public DynamicListInterface<capnp::DynamicList::Reader> {
	using DynamicListInterface::DynamicListInterface;
	
	DynamicListReader(DynamicListBuilder other);
};

struct DynamicListBuilder : public DynamicListInterface<capnp::DynamicList::Builder> {	
	using DynamicListInterface::DynamicListInterface;
	
	// Builder interface
	void set(uint32_t, py::object value);
	DynamicListBuilder initList(uint32_t idx, uint32_t size);
	
	static DynamicListBuilder cloneFrom(capnp::DynamicList::Reader reader);
};

struct DataReader : public WithMessage<capnp::Data::Reader> {
	using WithMessage::WithMessage;
	
	DataReader(DataBuilder);
	
	py::buffer_info buffer();
	kj::String repr();
	
	DataBuilder clone();

	static DataReader from(kj::Array<const byte>);
};

struct DataBuilder : public WithMessage<capnp::Data::Builder> {
	using WithMessage::WithMessage;
	
	py::buffer_info buffer();
	kj::String repr();
	
	DataBuilder clone();
	
	static DataBuilder from(kj::Array<byte>);
	static DataBuilder cloneFrom(kj::ArrayPtr<const byte>);
};

struct TextReader : public WithMessage<capnp::Text::Reader> {
	using WithMessage::WithMessage;
	
	TextReader(TextBuilder);
	
	kj::StringPtr repr();
	
	TextBuilder clone();
	
	static TextReader from(kj::String);
};

struct TextBuilder : public WithMessage<capnp::Text::Builder> {
	using WithMessage::WithMessage;
	
	kj::StringPtr repr();
	
	TextBuilder clone();
	
	static TextBuilder cloneFrom(kj::StringPtr);
};

struct AnyReader : public WithMessage<capnp::AnyPointer::Reader> {
	using WithMessage::WithMessage;
	
	AnyReader(AnyBuilder);
	
	kj::String repr();
	
	AnyBuilder clone();
};

struct AnyBuilder : public WithMessage<capnp::AnyPointer::Builder> {
	using WithMessage::WithMessage;
	
	kj::String repr();
	
	void setList(DynamicListReader);
	void setStruct(DynamicStructReader);
	void adopt(DynamicOrphan&);
	
	DynamicOrphan disown();
	
	AnyBuilder clone();
	
	static AnyBuilder cloneFrom(capnp::AnyPointer::Reader);
};

struct DynamicOrphan : public WithMessage<capnp::Orphan<capnp::DynamicValue>> {
	using WithMessage::WithMessage;
	
	bool consumed = false;
	
	DynamicValueBuilder get();
	capnp::Orphan<capnp::DynamicValue>&& release();
};

struct EnumInterface : public capnp::DynamicEnum {
	using DynamicEnum::DynamicEnum;
	
	inline EnumInterface(const capnp::DynamicEnum& other) : DynamicEnum(other) {}
	
	kj::String repr();
	bool eq1(DynamicEnum& other);
	bool eq2(uint16_t other);
	bool eq3(kj::StringPtr other);
};

//! Implements the descriptor protocol for Cap'n'Proto fields
struct FieldDescriptor : public capnp::StructSchema::Field {
	using Field::Field;
	
	inline FieldDescriptor(const Field& field) : Field(field) {}
	
	DynamicValueBuilder get1(DynamicStructBuilder&, py::object);
	DynamicValueReader get2(DynamicStructReader&, py::object);
	DynamicValuePipeline get3(DynamicStructPipeline&, py::object);
	kj::String get4(py::none, py::type);
	
	void set(DynamicStructBuilder obj, py::object value);
	void del(DynamicStructBuilder obj);
};

}