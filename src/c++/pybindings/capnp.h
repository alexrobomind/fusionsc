#pragma once

#include "fscpy.h"

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/schema.h>

namespace fscpy {

struct CapnpObject;
struct CapnpReader;
struct CapnpBuilder;

struct DynamicValueReader;
struct DynamicValueBuilder;
struct DynamicValuePipeline;

struct DynamicStructCommon;
struct DynamicStructReader;
struct DynamicStructBuilder;
struct DynamicStructPipeline;

struct DynamicListCommon;
struct DynamicListReader;
struct DynamicListBuilder;

struct DataCommon;
struct DataReader;
struct DataBuilder;

struct TextCommon;
struct TextReader;
struct TextBuilder; // Who the f'ck modifies a text in-place anyway?

struct AnyReader;
struct AnyBuilder;

struct DynamicCapabilityClient;
struct DynamicCapabilityServer;

struct DynamicCallContext;

// using DynamicCapabilityClient = capnp::DynamicCapability::Client;
// using DynamicCapabilityServer = capnp::DynamicCapability::Server;

struct EnumInterface;

struct DynamicOrphan;

struct FieldDescriptor;
struct ConstantValue;

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

// inline size_t dbgMessageCount = 0;

struct MessageHook : public kj::Refcounted {	
	inline MessageHook(Own<const void> data) : data(mv(data)) {
		// KJ_DBG("Message created", this, dbgMessageCount);
		// ++dbgMessageCount;
	}
	inline ~MessageHook() { /* --dbgMessageCount; KJ_DBG("Message deleted", this, dbgMessageCount);*/ }
	
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

struct CapnpObject {
	DynamicStructReader encodeSchema();
	
	virtual capnp::Type getType() = 0;
	inline virtual ~CapnpObject() noexcept(false) {};
};

struct CapnpReader {};
struct CapnpBuilder {};

struct DynamicValueReader : public WithMessage<capnp::DynamicValue::Reader>, CapnpReader {
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
	// COPY_WITHOUT_MSG(DynamicValueReader, DynamicCapabilityClient);
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

struct DynamicValueBuilder : public WithMessage<capnp::DynamicValue::Builder>, CapnpBuilder {
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
	// COPY_WITHOUT_MSG(DynamicValueBuilder, DynamicCapabilityClient);
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

struct DynamicStructCommon : public CapnpObject {
	~DynamicStructCommon() noexcept(false) = default;
};

template<typename StructType>
struct DynamicStructInterface : public WithMessage<StructType>, public DynamicStructCommon {
	using WithMessage<StructType>::WithMessage;
	
	capnp::Type getType() override;
	
	DynamicValueType<StructType> get(kj::StringPtr field);
	DynamicValueType<StructType> getCapnp(capnp::StructSchema::Field field);
	bool has(kj::StringPtr field);
	kj::StringPtr whichStr();
	
	kj::String repr();
	kj::String toYaml(bool flow);
	
	uint32_t size();
	uint64_t totalBytes();
	
	py::dict asDict();
	
	py::buffer_info buffer();
	
	py::bytes canonicalize();
	
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

struct DynamicStructReader : public DynamicStructInterface<capnp::DynamicStruct::Reader>, CapnpReader {
	using DynamicStructInterface::DynamicStructInterface;
	
	DynamicStructReader(DynamicStructBuilder other);
};

struct DynamicStructBuilder : public DynamicStructInterface<capnp::DynamicStruct::Builder>, CapnpBuilder {
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

struct DynamicListCommon : public CapnpObject {
	~DynamicListCommon() noexcept(false) = default;
};

template<typename ListType>
struct DynamicListInterface : public WithMessage<ListType>, DynamicListCommon {
	using WithMessage<ListType>::WithMessage;
	
	capnp::Type getType() override;
	
	DynamicValueType<ListType> get(int64_t idx);
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
	inline Iterator end() { return Iterator(*this, ListType::size()); };
	
	uint32_t preprocessIndex(int64_t);
};

struct DynamicListReader : public DynamicListInterface<capnp::DynamicList::Reader>, CapnpReader {
	using DynamicListInterface::DynamicListInterface;
	
	DynamicListReader(DynamicListBuilder other);
};

struct DynamicListBuilder : public DynamicListInterface<capnp::DynamicList::Builder>, CapnpBuilder {	
	using DynamicListInterface::DynamicListInterface;
	
	// Builder interface
	void set(int64_t, py::object value);
	DynamicListBuilder initList(int64_t idx, uint32_t size);
	
	static DynamicListBuilder cloneFrom(capnp::DynamicList::Reader reader);
};

struct DataCommon : public CapnpObject {
	capnp::Type getType() override;
	~DataCommon() noexcept(false) = default;
};

struct DataReader : public WithMessage<capnp::Data::Reader>, DataCommon, CapnpReader {
	using WithMessage::WithMessage;
	
	DataReader(DataBuilder);
	
	py::buffer_info buffer();
	kj::String repr();
	
	DataBuilder clone();

	static DataReader from(kj::Array<const byte>);
};

struct DataBuilder : public WithMessage<capnp::Data::Builder>, DataCommon, CapnpBuilder {
	using WithMessage::WithMessage;
	
	py::buffer_info buffer();
	kj::String repr();
	
	DataBuilder clone();
	
	static DataBuilder from(kj::Array<byte>);
	static DataBuilder cloneFrom(kj::ArrayPtr<const byte>);
};

struct TextCommon : public CapnpObject {
	capnp::Type getType() override;
	~TextCommon() noexcept(false) = default;
};

struct TextReader : public WithMessage<capnp::Text::Reader>, TextCommon, CapnpReader {
	using WithMessage::WithMessage;
	
	TextReader(TextBuilder);
	
	kj::StringPtr repr();
	
	TextBuilder clone();
	
	static TextReader from(kj::String);
};

struct TextBuilder : public WithMessage<capnp::Text::Builder>, TextCommon, CapnpBuilder {
	using WithMessage::WithMessage;
	
	kj::StringPtr repr();
	
	TextBuilder clone();
	
	static TextBuilder cloneFrom(kj::StringPtr);
};

struct AnyCommon : public CapnpObject {
	capnp::Type getType() override;
	~AnyCommon() noexcept(false) = default;
};

struct AnyReader : public WithMessage<capnp::AnyPointer::Reader>, AnyCommon, CapnpReader {
	using WithMessage::WithMessage;
	
	AnyReader(AnyBuilder);
	
	kj::String repr();
	
	AnyBuilder clone();
	DynamicValueReader interpretAs(capnp::Type);
};

struct AnyBuilder : public WithMessage<capnp::AnyPointer::Builder>, AnyCommon, CapnpBuilder {
	using WithMessage::WithMessage;
	
	kj::String repr();
	
	void setList(DynamicListReader);
	void setStruct(DynamicStructReader);
	void setCap(DynamicCapabilityClient);
	void adopt(DynamicOrphan&);
	
	DynamicOrphan disown();
	
	AnyBuilder clone();
	
	DynamicValueBuilder interpretAs(capnp::Type);
	DynamicValueBuilder initBuilderAs(capnp::Type, uint32_t size);
	DynamicValueBuilder assignAs(DynamicValueReader);
	
	static AnyBuilder cloneFrom(capnp::AnyPointer::Reader);
};

struct DynamicCapabilityClient : public capnp::DynamicCapability::Client, CapnpObject {
	using Client::Client;
	
	inline DynamicCapabilityClient(Client& c) : Client(c) {}
	inline DynamicCapabilityClient(Client&& c) : Client(mv(c)) {}
	
	inline DynamicCapabilityClient(DynamicCapabilityClient& o) :
		Client::Client(o), myExecutor(o.executor().addRef())
	{}
	
	DynamicCapabilityClient(DynamicCapabilityClient&&) = default;
	
	capnp::Type getType() override;
	
	inline bool locallyOwned() { return &(executor()) == &(kj::getCurrentThreadExecutor()); }
	const kj::Executor& executor() { return *(myExecutor.get()); }
	
private:
	Own<const kj::Executor> myExecutor = kj::getCurrentThreadExecutor().addRef();
};

struct DynamicCallContext {
	using WrappedContext = capnp::CallContext<capnp::DynamicStruct, capnp::DynamicStruct>;
	WrappedContext backend;
	
	DynamicCallContext(WrappedContext wc);
	
	DynamicStructReader getParams();
	
	DynamicStructBuilder initResults();
	DynamicStructBuilder getResults();
	void setResults(DynamicStructReader result);
	
	Promise<void> tailCall(DynamicCapabilityClient, kj::StringPtr, DynamicStructReader);
};

struct DynamicCapabilityServer {
	inline DynamicCapabilityServer(capnp::InterfaceSchema schema ) : mySchema(schema) {}
	inline capnp::InterfaceSchema& schema() { return mySchema; }
	
	DynamicCapabilityClient thisCap();
	
	// This is public because other classes need to manipulate it
	capnp::ClientHook* activeClient = nullptr;
private:
	capnp::InterfaceSchema mySchema;
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
	bool eq1(EnumInterface& other);
	bool eq2(uint16_t other);
	bool eq3(kj::StringPtr other);
};

//! Implements the descriptor protocol for Cap'n'Proto fields
struct FieldDescriptor : public capnp::StructSchema::Field {
	// using Field::Field;
	
	inline FieldDescriptor(const Field& field, kj::String docString) : Field(field), docString(mv(docString)) {}
	
	DynamicValueBuilder get1(DynamicStructBuilder&, py::object);
	DynamicValueReader get2(DynamicStructReader&, py::object);
	DynamicValuePipeline get3(DynamicStructPipeline&, py::object);
	FieldDescriptor& get4(py::none, py::type);
	
	void set(DynamicStructBuilder obj, py::object value);
	void del(DynamicStructBuilder obj);
	
	py::object doc();
	
	kj::String docString;
};

struct ConstantValue {
	capnp::ConstSchema schema;
	
	inline ConstantValue(capnp::ConstSchema schema) :
		schema(mv(schema))
	{}
	
	inline DynamicValueReader value() {
		return DynamicValueReader(noMessage(capnp::DynamicValue::Reader(schema)));
	}
	
	inline capnp::Type type() {
		return schema.getType();
	}
};

struct MethodInfo {
	capnp::InterfaceSchema::Method method;
	
	inline MethodInfo(capnp::InterfaceSchema::Method m) :
		method(m)
	{}

	inline capnp::StructSchema paramType() {
		return method.getParamType();
	}
	
	inline capnp::StructSchema resultType() {
		return method.getResultType();
	}
	
	inline uint16_t getOrdinal() {
		return method.getOrdinal();
	}
};

DynamicValueBuilder castBuilder(DynamicValueBuilder, capnp::Type);
DynamicValueReader castReader(DynamicValueReader, capnp::Type);

kj::String getJsonSchemaForType(capnp::Type);

}
