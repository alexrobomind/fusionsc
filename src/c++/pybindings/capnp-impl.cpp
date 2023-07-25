#include "fscpy.h"

#include "assign.h"
#include "tensor.h"

#include <kj/encoding.h>

#include <fsc/yaml.h>

using capnp::AnyPointer;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicCapability;
using capnp::DynamicEnum;
using capnp::DynamicList;

namespace fscpy {

namespace {	
	template<typename T>
	auto adjust(capnp::Type type, T&& target) {
		// Fields of constrained AnyPointer type need to be adjusted
		if(type.isAnyPointer() && type.whichAnyPointerKind() == capnp::schema::Type::AnyPointer::Unconstrained::CAPABILITY) {
			auto schema = defaultLoader.importBuiltin<capnp::Capability>();
			
			auto asAny = target.template as<capnp::AnyPointer>();
			target = asAny.template getAs<capnp::DynamicCapability>(schema.asInterface());
		}
		
		return target;
	}
}
	
// ------------------------------------------------- Message --------------------------------------------------

// All inline

// ----------------------------------------------- DynamicValue -----------------------------------------------

// DynamicValueReader

DynamicValueReader::DynamicValueReader(DynamicValueBuilder& other) :
	WithMessage(kj::Own<const void>(), capnp::Void())
{
	using DV = capnp::DynamicValue;
	
	#define HANDLE_CASE(branch, Type) \
		case DV::branch: \
			*this = other.as<Type>(); \
			return;
	
	switch(other.getType()) {
		HANDLE_CASE(VOID, capnp::Void)
		HANDLE_CASE(BOOL, bool)
		HANDLE_CASE(INT, int64_t)
		HANDLE_CASE(UINT, uint64_t)
		HANDLE_CASE(FLOAT, double)
		HANDLE_CASE(CAPABILITY, capnp::DynamicCapability)
		HANDLE_CASE(ENUM, capnp::DynamicEnum)
		
		case DV::TEXT:
			*this = TextReader(other.asText());
			return;
		case DV::DATA:
			*this = DataReader(other.asData());
			return;
		case DV::LIST:
			*this = DynamicListReader(other.asList());
			return;
		case DV::STRUCT:
			*this = DynamicStructReader(other.asStruct());
			return;
		case DV::ANY_POINTER:
			*this = AnyReader(other.asAny());
			return;
	}
	
	KJ_UNREACHABLE;
	
	#undef HANDLE_CASE
}

DynamicStructReader DynamicValueReader::asStruct() {
	return DynamicStructReader(shareMessage(*this), as<DynamicStruct>());
}

DynamicListReader DynamicValueReader::asList() {
	return DynamicListReader(shareMessage(*this), as<DynamicList>());
}

DataReader DynamicValueReader::asData() {
	return DataReader(shareMessage(*this), as<capnp::Data>());
}

TextReader DynamicValueReader::asText() {
	return TextReader(shareMessage(*this), as<capnp::Text>());
}

AnyReader DynamicValueReader::asAny() {
	return AnyReader(shareMessage(*this), as<AnyPointer>());
}

EnumInterface DynamicValueReader::asEnum() {
	return as<capnp::DynamicEnum>();
}

DynamicValueBuilder DynamicValueReader::clone() {
	return DynamicValueBuilder::cloneFrom(*this);
}

// DynamicValueBuilder

DynamicStructBuilder DynamicValueBuilder::asStruct() {
	return DynamicStructBuilder(shareMessage(*this), as<DynamicStruct>());
}

DynamicListBuilder DynamicValueBuilder::asList() {
	return DynamicListBuilder(shareMessage(*this), as<DynamicList>());
}

DataBuilder DynamicValueBuilder::asData() {
	return DataBuilder(shareMessage(*this), as<capnp::Data>());
}

TextBuilder DynamicValueBuilder::asText() {
	return TextBuilder(shareMessage(*this), as<capnp::Text>());
}

AnyBuilder DynamicValueBuilder::asAny() {
	return AnyBuilder(shareMessage(*this), as<AnyPointer>());
}

EnumInterface DynamicValueBuilder::asEnum() {
	return as<capnp::DynamicEnum>();
}

DynamicValueBuilder DynamicValueBuilder::clone() {
	return DynamicValueBuilder::cloneFrom(toReader(wrapped()));
}

DynamicValueBuilder DynamicValueBuilder::cloneFrom(capnp::DynamicValue::Reader reader) {
	using DV = capnp::DynamicValue;
	
	#define HANDLE_CASE(branch, Type) \
		case DV::branch: \
			return DynamicValueBuilder(kj::attachRef(ANONYMOUS), reader.as<Type>());
	
	switch(reader.getType()) {
		HANDLE_CASE(VOID, capnp::Void)
		HANDLE_CASE(BOOL, bool)
		HANDLE_CASE(INT, int64_t)
		HANDLE_CASE(UINT, uint64_t)
		HANDLE_CASE(FLOAT, double)
		HANDLE_CASE(CAPABILITY, capnp::DynamicCapability)
		HANDLE_CASE(ENUM, capnp::DynamicEnum)
		
		case DV::TEXT:
			return TextBuilder::cloneFrom(reader.as<capnp::Text>());
		case DV::DATA:
			return DataBuilder::cloneFrom(reader.as<capnp::Data>());
		case DV::LIST:
			return DynamicListBuilder::cloneFrom(reader.as<capnp::DynamicList>());
		case DV::STRUCT:
			return DynamicStructBuilder::cloneFrom(reader.as<capnp::DynamicStruct>());
		case DV::ANY_POINTER:
			return AnyBuilder::cloneFrom(reader.as<capnp::AnyPointer>());
	}
	
	KJ_UNREACHABLE;
	
	#undef HANDLE_CASE
}

// DynamicValuePipeline

DynamicStructPipeline DynamicValuePipeline::asStruct() {
	return DynamicStructPipeline(
		typeless.noop(), schema.asStruct()
	);
}

capnp::DynamicCapability::Client DynamicValuePipeline::asCapability() {
	capnp::Capability::Client anyCap = typeless;
	return anyCap.castAs<DynamicCapability>(schema.asInterface());
}

// ----------------------------------------------- DynamicStruct -----------------------------------------------

// DynamicStructInterface

template<typename StructType>
DynamicValueType<StructType> DynamicStructInterface<StructType>::get(kj::StringPtr field) {
	return getCapnp(this -> getSchema().getFieldByName(field));
}

template<typename StructType>
DynamicValueType<StructType> DynamicStructInterface<StructType>::getCapnp(capnp::StructSchema::Field field) {
	return DynamicValueType<StructType>(shareMessage(*this), adjust(field.getType(), this -> wrapped().get(field)));
}

template<typename StructType>
bool DynamicStructInterface<StructType>::has(kj::StringPtr field) {
	return this -> wrapped().has(field);
}

template<typename StructType>
kj::StringPtr DynamicStructInterface<StructType>::whichStr() {
	KJ_IF_MAYBE(pWhich, this -> which()) {
		return pWhich -> getProto().getName();
	}
	
	return "";
}

template<typename StructType>
kj::String DynamicStructInterface<StructType>::repr() {
	return toYaml(false);
}

template<typename StructType>
kj::String DynamicStructInterface<StructType>::toYaml(bool flow) {
	auto emitter = kj::heap<YAML::Emitter>();
	
	if(flow) {
		emitter -> SetMapFormat(YAML::Flow);
		emitter -> SetSeqFormat(YAML::Flow);
	}
	
	(*emitter) << toReader(this -> wrapped());
	
	kj::ArrayPtr<char> stringData(const_cast<char*>(emitter -> c_str()), emitter -> size() + 1);
	
	return kj::String(stringData.attach(mv(emitter)));
}

template<typename StructType>
uint32_t DynamicStructInterface<StructType>::size() {
	auto schema = this -> getSchema();
	size_t result = schema.getNonUnionFields().size();
	if(schema.getUnionFields().size() > 0)
		result += 1;
	
	return result;
}

template<typename StructType>
uint64_t DynamicStructInterface<StructType>::totalBytes() {
	return this -> wrapped().totalSize().wordCount * 8;
}

template<typename StructType>
py::dict DynamicStructInterface<StructType>::asDict() {
	py::dict result;
	
	auto schema = this -> getSchema();
	
	for(auto field : schema.getNonUnionFields()) {
		result[field.getProto().getName().cStr()] = getCapnp(field);
	}
	
	KJ_IF_MAYBE(pField, this -> which()) {
		result[pField -> getProto().getName().cStr()] = getCapnp(*pField);
	}
	
	return result;
}

template<typename StructType>
py::buffer_info DynamicStructInterface<StructType>::buffer() {
	return getTensor(*this);
}

template<typename StructType>
DynamicStructBuilder DynamicStructInterface<StructType>::clone() {
	return DynamicStructBuilder::cloneFrom(toReader(this -> wrapped()));
}

template class DynamicStructInterface<capnp::DynamicStruct::Reader>;
template class DynamicStructInterface<capnp::DynamicStruct::Builder>;

// DynamicStructReader (nothing required)

DynamicStructReader::DynamicStructReader(DynamicStructBuilder other) :
	DynamicStructReader(shareMessage(other), other.asReader())
{}

// DynamicStructBuilder

void DynamicStructBuilder::set(kj::StringPtr field, py::object value) {
	// Delegate to assignment protocol
	assign(*this, field, mv(value));
}

DynamicValueBuilder DynamicStructBuilder::init(kj::StringPtr field) {
	return DynamicValueBuilder(shareMessage(*this), wrapped().init(field));
}

DynamicListBuilder DynamicStructBuilder::initList(kj::StringPtr field, uint32_t size) {
	return DynamicListBuilder(shareMessage(*this), wrapped().init(field, size).as<capnp::DynamicList>());
}


DynamicOrphan DynamicStructBuilder::disown(kj::StringPtr field) {
	return DynamicOrphan(shareMessage(*this), wrapped().disown(field));
}

DynamicStructBuilder DynamicStructBuilder::cloneFrom(capnp::DynamicStruct::Reader reader) {
	auto msg = kj::heap<capnp::MallocMessageBuilder>();
	msg -> setRoot(reader);
	
	auto root = msg -> getRoot<DynamicStruct>(reader.getSchema());
	return DynamicStructBuilder(mv(msg), root);
}

// DynamicStructPipeline

DynamicValuePipeline DynamicStructPipeline::get(kj::StringPtr fieldName) {
	return getCapnp(schema.getFieldByName(fieldName));
}

DynamicValuePipeline DynamicStructPipeline::getCapnp(capnp::StructSchema::Field field) {
	capnp::AnyPointer::Pipeline typelessValue(nullptr);
	
	KJ_REQUIRE(field.getProto().getDiscriminantValue() == static_cast<uint16_t>(-1), "Can not access union fields in pipelines");
	
	// Groups must point to the same field
	if(field.getProto().isGroup()) {
		typelessValue = typeless.noop();
	} else {
		KJ_ASSERT(field.getProto().isSlot());
		
		auto asSlot = field.getProto().getSlot();
		capnp::AnyStruct::Pipeline typelessAsAnyStruct(typeless.noop());
		typelessValue = typelessAsAnyStruct.getPointerField(asSlot.getOffset());
	}
	
	auto fieldType = field.getType();
	
	KJ_REQUIRE(fieldType.isStruct() || fieldType.isInterface() || fieldType.isAnyPointer());
	
	if(fieldType.isStruct()) {
		return DynamicValuePipeline(
			mv(typelessValue), field.getType().asStruct()
		);
	} else if(fieldType.isInterface()) {
		return DynamicValuePipeline(
			mv(typelessValue), field.getType().asInterface()
		);
	} else {
		// AnyStruct fields are not accessible
		KJ_REQUIRE(fieldType.whichAnyPointerKind() == capnp::schema::Type::AnyPointer::Unconstrained::CAPABILITY);
		
		auto schema = defaultLoader.importBuiltin<capnp::Capability>().asInterface();
		return DynamicValuePipeline(
			mv(typelessValue), mv(schema)
		);
	}
}


// ----------------------------------------------- DynamicList -----------------------------------------------

// DynamicListInterface

template<typename ListType>
DynamicValueType<ListType> DynamicListInterface<ListType>::get(uint32_t idx) {	
	return DynamicValueType<ListType>(shareMessage(*this), adjust(this -> getSchema().getElementType(), this -> wrapped()[idx]));
}

template<typename ListType>
py::buffer_info DynamicListInterface<ListType>::buffer() {
	// Extract raw data
	kj::ArrayPtr<const byte> rawBytes = capnp::AnyList::Reader(toReader(this -> wrapped())).getRawBytes();
	byte* bytesPtr = const_cast<byte*>(rawBytes.begin());
	
	// Read format
	capnp::ListSchema schema = this -> getSchema();
	auto format = pyFormat(schema.getElementType());
	size_t elementSize = kj::get<0>(format);
	kj::StringPtr formatString = kj::get<1>(format);
	
	// Sanity checks
	KJ_REQUIRE(elementSize * this -> size() == rawBytes.size());
			
	return py::buffer_info(
		(void*) bytesPtr, elementSize, std::string(formatString.cStr()), this -> size(), !isBuilder<ListType>()
	);
}

template<typename ListType>
DynamicListBuilder DynamicListInterface<ListType>::clone() {
	return DynamicListBuilder::cloneFrom(toReader(this -> wrapped()));
}

template class DynamicListInterface<capnp::DynamicList::Reader>;
template class DynamicListInterface<capnp::DynamicList::Builder>;

// DynamicListReader

DynamicListReader::DynamicListReader(DynamicListBuilder other) :
	DynamicListReader(shareMessage(other), other.asReader())
{}

// DynamicListBuilder

void DynamicListBuilder::set(uint32_t idx, py::object value) {
	assign(*this, idx, mv(value));
}

DynamicListBuilder DynamicListBuilder::initList(uint32_t idx, uint32_t size) {
	return DynamicListBuilder(shareMessage(*this), wrapped().init(idx, size).as<capnp::DynamicList>());
}

DynamicListBuilder DynamicListBuilder::cloneFrom(capnp::DynamicList::Reader reader) {
	auto msg = kj::heap<capnp::MallocMessageBuilder>();
	msg -> setRoot(reader);
	
	auto root = msg -> getRoot<DynamicList>(reader.getSchema());
	return DynamicListBuilder(mv(msg), root);
}

// ----------------------------------------------- Data -----------------------------------------------

// DataReader

DataReader::DataReader(DataBuilder other) :
	DataReader(shareMessage(other), other.asReader())
{}

py::buffer_info DataReader::buffer() {
	return py::buffer_info(begin(), size(), true);
}

kj::String DataReader::repr() {
	return kj::encodeBase64(wrapped());
}

DataReader DataReader::from(kj::Array<const byte> arr) {
	auto ptr = arr.asPtr();
	return DataReader(kj::heap(mv(arr)), ptr);
}

DataBuilder DataReader::clone() {
	return DataBuilder::cloneFrom(*this);
}

// DataBuilder

py::buffer_info DataBuilder::buffer() {
	return py::buffer_info(begin(), size(), false);
}

kj::String DataBuilder::repr() {
	return kj::encodeBase64(asReader());
}

DataBuilder DataBuilder::clone() {
	return DataBuilder::cloneFrom(*this);
}

DataBuilder DataBuilder::from(kj::Array<byte> arr) {
	auto ptr = arr.asPtr();
	return DataBuilder(kj::heap(mv(arr)), ptr);
}

DataBuilder DataBuilder::cloneFrom(kj::ArrayPtr<const byte> ref) {
	return DataBuilder::from(kj::heapArray(ref));
}

// ----------------------------------------------- Text -----------------------------------------------

// TextReader

TextReader::TextReader(TextBuilder other) :
	TextReader(shareMessage(other), other.asReader())
{}

kj::StringPtr TextReader::repr() {
	return *this;
}

TextReader TextReader::from(kj::String s) {
	auto ptr = s.asPtr();
	return TextReader(kj::heap(mv(s)), ptr);
}

TextBuilder TextReader::clone() {
	return TextBuilder::cloneFrom(*this);
}

// TextBuilder

kj::StringPtr TextBuilder::repr() {
	return *this;
}

TextBuilder TextBuilder::cloneFrom(kj::StringPtr ptr) {
	auto asArray = kj::heapString(ptr).releaseArray();
	auto asPtr = asArray.asPtr();
	
	return TextBuilder(kj::heap(mv(asArray)), asPtr.begin(), asPtr.size());
}

TextBuilder TextBuilder::clone() {
	return TextBuilder::cloneFrom(*this);
}

// ----------------------------------------------- AnyPointer -----------------------------------------

// AnyReader

AnyReader::AnyReader(AnyBuilder other) :
	AnyReader(shareMessage(other), other.asReader())
{}

kj::String AnyReader::repr() {
	return kj::str("<opaque pointer>");
}

AnyBuilder AnyReader::clone() {
	return AnyBuilder::cloneFrom(*this);
}

// AnyBuilder

kj::String AnyBuilder::repr() {
	return kj::str("<opaque pointer>");
}

void AnyBuilder::setList(DynamicListReader r) {
	setAs<DynamicList>(r.wrapped());
}

void AnyBuilder::setStruct(DynamicStructReader r) {
	setAs<DynamicStruct>(r.wrapped());
}

void AnyBuilder::adopt(DynamicOrphan& orphan) {
	wrapped().adopt(orphan.release());
}

DynamicOrphan AnyBuilder::disown() {
	return DynamicOrphan(shareMessage(*this), wrapped().disown());
}

AnyBuilder AnyBuilder::cloneFrom(capnp::AnyPointer::Reader ptr) {
	auto msg = kj::heap<capnp::MallocMessageBuilder>();
	msg -> setRoot(ptr);
	
	auto root = msg -> getRoot<capnp::AnyPointer>();
	
	return AnyBuilder(mv(msg), root);
}

AnyBuilder AnyBuilder::clone() {
	return AnyBuilder::cloneFrom(*this);
}

// DynamicOrphan

DynamicValueBuilder DynamicOrphan::get() {
	KJ_REQUIRE(!consumed, "Can not access contents of adopted orphan");
	return DynamicValueBuilder(shareMessage(*this), wrapped().get());
}

capnp::Orphan<capnp::DynamicValue>&& DynamicOrphan::release() {
	KJ_REQUIRE(!consumed, "Orphan was already adopted");
	return mv(wrapped());
}

// ------------------------------------------------ Enums -----------------------------------------

kj::String EnumInterface::repr() {
	KJ_IF_MAYBE(pEnumerant, getEnumerant()) {
		return kj::str("'", pEnumerant -> getProto().getName(), "'");
	}
	return kj::str("<unknown> (", getRaw(), ")");
}

bool EnumInterface::eq1(DynamicEnum& other) {
	return getSchema() == other.getSchema() && getRaw() == other.getRaw();
}

bool EnumInterface::eq2(uint16_t other) {
	return getRaw() == other;
}

bool EnumInterface::eq3(kj::StringPtr other) {
	KJ_IF_MAYBE(pEnumerant, getEnumerant()) {
		return pEnumerant -> getProto().getName() == other;
	}
	return false;
}

// ----------------------------------------------- Fields -----------------------------------------

// FieldDescriptor

DynamicValueBuilder FieldDescriptor::get1(DynamicStructBuilder& builder, py::object) {
	return builder.get(getProto().getName());
}

DynamicValueReader FieldDescriptor::get2(DynamicStructReader& reader, py::object) {
	return reader.get(getProto().getName());
}

DynamicValuePipeline FieldDescriptor::get3(DynamicStructPipeline& reader, py::object) {
	return reader.get(getProto().getName());
}

kj::String FieldDescriptor::get4(py::none, py::type type) {
	return kj::str("Field ", getProto().getName(), " of class ", py::cast<kj::StringPtr>(py::str(type)));
}

void FieldDescriptor::set(DynamicStructBuilder obj, py::object value) {
	obj.set(getProto().getName(), value);
}
void FieldDescriptor::del(DynamicStructBuilder obj) {
	obj.wrapped().init(getProto().getName());
}

};