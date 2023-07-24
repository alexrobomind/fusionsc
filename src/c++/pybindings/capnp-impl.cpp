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
			
			auto asAny = target.as<capnp::AnyPointer>();
			target = asAny.getAs<capnp::DynamicCapability>(schema.asInterface());
		}
		
		return target;
	}
}
	
// ------------------------------------------------- Message --------------------------------------------------

// All inline

// ----------------------------------------------- DynamicValue -----------------------------------------------

// DynamicValueReader

DynamicStructReader DynamicValueReader::asStruct() {
	return DynamicStructReader(*this, as<DynamicStruct>());
}

DynamicListReader DynamicValueReader::asList() {
	return DynamicListReader(*this, as<DynamicList>());
}

DataReader DynamicValueReader::asData() {
	return DataReader(*this, as<capnp::Data>());
}

TextReader DynamicValueReader::asText() {
	return TextReader(*this, as<capnp::Text>());
}

AnyReader DynamicValueReader::asAny() {
	return AnyReader(*this, as<AnyPointer>());
}

// DynamicValueBuilder

DynamicStructBuilder DynamicValueBuilder::asStruct() {
	return DynamicStructBuilder(*this, as<DynamicStruct>());
}

DynamicListBuilder DynamicValueBuilder::asList() {
	return DynamicListBuilder(*this, as<DynamicList>());
}

DataBuilder DynamicValueBuilder::asData() {
	return DataBuilder(*this, as<capnp::Data>());
}

TextBuilder DynamicValueBuilder::asText() {
	return TextBuilder(*this, as<capnp::Text>());
}

AnyBuilder DynamicValueBuilder::asAny() {
	return AnyBuilder(*this, as<AnyPointer>());
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
	return getCapnp(getSchema().getFieldByName(field));
}

template<typename StructType>
DynamicValueType<StructType> DynamicStructInterface<StructType>::getCapnp(capnp::StructSchema::Field field) {
	return DynamicValueType<StructType>(*this, adjust(field.getType(), wrapped().get(field)));
}

template<typename StructType>
bool DynamicStructInterface<StructType>::has(kj::StringPtr field) {
	return wrapped().has(field);
}

template<typename StructType>
kj::StringPtr DynamicStructInterface<StructType>::whichStr() {
	KJ_IF_MAYBE(pWhich, which()) {
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
	
	(*emitter) << toReader(wrapped());
	
	kj::ArrayPtr<char> stringData(const_cast<char*>(emitter -> c_str()), emitter -> size() + 1);
	
	return kj::String(stringData.attach(mv(emitter)));
}

template<typename StructType>
uint32_t DynamicStructInterface<StructType>::size() {
	auto schema = getSchema();
	size_t result = schema.getNonUnionFields().size();
	if(schema.getUnionFields().size() > 0)
		result += 1;
	
	return result;
}

template<typename StructType>
py::dict DynamicStructInterface<StructType>::asDict() {
	py::dict result;
	
	auto schema = getSchema();
	
	for(auto field : schema.getNonUnionFields()) {
		result[field.getProto().getName().cStr()] = getCapnp(field);
	}
	
	KJ_IF_MAYBE(pField, which()) {
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
	auto msg = kj::heap<capnp::MallocMessageBuilder>();
	msg -> setRoot(toReader(wrapped()));
	
	return DynamicStructBuilder(mv(msg), msg -> getRoot<DynamicStruct>(getSchema()));
}

template class DynamicStructInterface<capnp::DynamicStruct::Reader>;
template class DynamicStructInterface<capnp::DynamicStruct::Builder>;

// DynamicStructReader (nothing required)

// DynamicStructBuilder

void DynamicStructBuilder::set(kj::StringPtr field, py::object value) {
	// Delegate to assignment protocol
	assign(*this, field, mv(value));
}

void DynamicStructBuilder::initList(kj::StringPtr field, uint32_t size) {
	init(field, size);
}

DynamicOrphan DynamicStructBuilder::disown(kj::StringPtr field) {
	return DynamicOrphan(*this, wrapped().disown(field));
}

// DynamicStructPipeline

DynamicValuePipeline DynamicStructPipeline::get(kj::StringPtr fieldName) {
	return getCapnp(schema.getFieldByName(fieldName));
}

DynamicValuePipeline DynamicStructPipeline::getCapnp(capnp::StructSchema::Field field) {
	capnp::AnyPointer::Pipeline typelessValue(nullptr);
	
	KJ_REQUIRE(field.getProto().getDiscriminantValue() == capnp::schema::Field::NO_DISCRIMINANT, "Can not access union fields in pipelines");
	
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
	return DynamicValueType<ListType>(*this, adjust(getSchema().getElementType(), wrapped()[idx]));
}

template<typename ListType>
py::buffer_info DynamicListInterface<ListType>::buffer() {
	// Extract raw data
	kj::ArrayPtr<const byte> rawBytes = capnp::AnyList::Reader(toReader(wrapped())).getRawBytes();
	byte* bytesPtr = const_cast<byte*>(rawBytes.begin());
	
	// Read format
	capnp::ListSchema schema = getSchema();
	auto format = pyFormat(schema.getElementType());
	size_t elementSize = kj::get<0>(format);
	kj::StringPtr formatString = kj::get<1>(format);
	
	// Sanity checks
	KJ_REQUIRE(elementSize * size() == rawBytes.size());
			
	return py::buffer_info(
		(void*) bytesPtr, elementSize, std::string(formatString.cStr()), size(), !isBuilder<ListType>()
	);
}

template<typename ListType>
DynamicListBuilder DynamicListInterface<ListType>::clone() {
	auto msg = kj::heap<capnp::MallocMessageBuilder>();
	msg -> setRoot(toReader(wrapped()));
	
	return DynamicListBuilder(mv(msg), msg -> getRoot<DynamicList>(getSchema()));
}

template class DynamicListInterface<capnp::DynamicList::Reader>;
template class DynamicListInterface<capnp::DynamicList::Builder>;

// DynamicListReader

// DynamicListBuilder

void DynamicListBuilder::set(uint32_t idx, py::object value) {
	assign(*this, idx, mv(value));
}

void DynamicListBuilder::initList(uint32_t idx, uint32_t size) {
	init(idx, size);
}

// ----------------------------------------------- Data -----------------------------------------------

// DataReader

py::buffer_info DataReader::buffer() {
	return py::buffer_info(begin(), size(), true);
}

kj::String DataReader::repr() {
	return kj::encodeBase64(wrapped());
}

// DataBuilder

py::buffer_info DataBuilder::buffer() {
	return py::buffer_info(begin(), size(), false);
}

kj::String DataBuilder::repr() {
	return kj::encodeBase64(asReader());
}

// ----------------------------------------------- Text -----------------------------------------------

kj::StringPtr TextReader::repr() {
	return *this;
}

// ----------------------------------------------- AnyPointer -----------------------------------------

// AnyReader

kj::String AnyReader::repr() {
	return kj::str("<opaque pointer>");
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
	return DynamicOrphan(*this, wrapped().disown());
}

// DynamicOrphan

DynamicValueBuilder DynamicOrphan::get() {
	KJ_REQUIRE(!consumed, "Can not access contents of adopted orphan");
	return DynamicValueBuilder(*this, wrapped().get());
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

DynamicValueBuilder FieldDescriptor::get1(DynamicStructBuilder builder) {
	return builder.get(getProto().getName());
}

DynamicValueReader FieldDescriptor::get2(DynamicStructReader reader) {
	return reader.get(getProto().getName());
}

kj::String FieldDescriptor::get3(py::none, py::type type) {
	return kj::str("Field ", getProto().getName(), " of class ", py::cast<kj::StringPtr>(py::str(type)));
}

void FieldDescriptor::set(DynamicStructBuilder obj, py::object value) {
	obj.set(getProto().getName(), value);
}
void FieldDescriptor::del(DynamicStructBuilder obj) {
	obj.wrapped().init(getProto().getName());
}

};