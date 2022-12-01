#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>

#include <kj/async.h>
#include <kj/mutex.h>

#include <capnp/dynamic.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/blob.h>
#include <capnp/endian.h>
#include <capnp/pretty-print.h>

#include <fsc/local.h>
#include <fsc/common.h>
#include <fsc/data.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>

#include "fscpy.h"
#include "loader.h"

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

namespace py = pybind11;

using namespace fscpy;

// Definitions local to translation unit

namespace {

/**
 *  Retrieves the values of a field and, if neccessary, creates
 *  a keep-alive reference to the parent (not all python types need
 *  that, and out of those that don't, some do not support custom
 *  attributes or weakrefs.
 */
py::object getField(py::object self, py::object field) {
	py::tuple getResult = self.attr("_get")(field);
	
	py::object fieldValue  = getResult[py::cast(0)];
	bool backReference = py::cast<bool>(getResult[1]);
	
	if(backReference)
		fieldValue.attr("_parent") = self;
	
	return fieldValue;
}

/**
 * Determines which readers / builders need to store a reference to
 * their parent to keep their data alive.
 */
template<typename T>
bool needsBackReference(T&& t) {
	switch(t.getType()) {
		case DynamicValue::TEXT:
		case DynamicValue::DATA:
		case DynamicValue::LIST:
		case DynamicValue::STRUCT:
		case DynamicValue::ANY_POINTER:
		case DynamicValue::CAPABILITY:
			return true;
		
		default:
			return false;
	};
}

//! Pipelines have their own keep-alive through PipelineHooks
template<>
bool needsBackReference<DynamicValuePipeline>(DynamicValuePipeline&& t) {
	return false;
}

//! Pipelines have their own keep-alive through PipelineHooks
template<>
bool needsBackReference<DynamicValuePipeline&>(DynamicValuePipeline& t) {
	return false;
}

/**
 * Implementation for _get used by getField, which determines, based on type
 * dispatch, whether the returned value needs a back-reference.
 */
template<typename T, typename Field>
py::tuple underscoreGet(T& ds, Field field) {
	auto cppValue = ds.get(field);
	bool nbr = needsBackReference(cppValue);
	
	// We REALLY don't want pybind11 to try to copy this
	// auto pCppValue = new decltype(cppValue)(mv(cppValue));
	
	return py::make_tuple(
		py::cast(mv(cppValue)),
		py::cast(nbr)
	);
};

//TODO: Buffer view for data

//! Binds capnp::Data::{Reader, Builder, Pipeline} and capnp::Text::{Reader, Builder, Pipeline}
void bindBlobClasses(py::module_& m) {
	using TR = capnp::Text::Reader;
	using TB = capnp::Text::Builder;
	using TP = capnp::Text::Pipeline;
	
	py::class_<TR, kj::StringPtr>(m, "TextReader", py::dynamic_attr());
	py::class_<TB>(m, "TextBuilder", py::dynamic_attr())
		.def("__repr__", [](TB& self) { return self.asString(); })
	;
	py::class_<TP>(m, "TextPipeline", py::dynamic_attr());
	
	using DR = capnp::Data::Reader;
	using DB = capnp::Data::Builder;
	using DP = capnp::Data::Pipeline;
	
	py::class_<DR>(m, "DataReader", py::dynamic_attr());
	py::class_<DB>(m, "DataBuilder", py::dynamic_attr());
	py::class_<DP>(m, "DataPipeline", py::dynamic_attr());
}

template<typename List>
struct Iterator {
	List& list;
	size_t pos;
	
	Iterator(List& list, size_t pos) : list(list), pos(pos) {}
	
	bool operator==(Iterator other) { return pos == other.pos; }
	bool operator!=(Iterator other) { return pos != other.pos; }
	Iterator& operator++() { ++pos; return *this; }
	
	auto operator*() { return list[pos]; }
};

template<typename T>
void defGetItemAndLen(py::class_<T>& c) {
	c.def("__getitem__", [](T& list, size_t idx) { return list[idx]; });
	c.def("__len__", [](T& list) { return list.size(); });
	c.def("__iter__",
		[](T& list) {
			//TODO: This relies on internals, but the default way looks bugged
			
			Iterator begin(list, 0);
			Iterator end(list, list.size());
			
			return py::make_iterator(begin, end);
		},
		py::keep_alive<0, 1>()
	);
}

bool isObjectType(capnp::Type type) {
	switch(type.which()) {
		case capnp::schema::Type::VOID:
		case capnp::schema::Type::BOOL:
		case capnp::schema::Type::TEXT:
		case capnp::schema::Type::DATA:
		case capnp::schema::Type::LIST:
		case capnp::schema::Type::STRUCT:
		case capnp::schema::Type::INTERFACE:
		case capnp::schema::Type::ANY_POINTER:
		case capnp::schema::Type::ENUM:
			return true;
		
		default:
			return false;
	}
}
	

Tuple<size_t, kj::StringPtr> pyFormat(capnp::Type type) {
	size_t elementSize = 0;
	kj::StringPtr formatString;
	
	switch(type.which()) {
		case capnp::schema::Type::VOID:
		case capnp::schema::Type::TEXT:
		case capnp::schema::Type::DATA:
		case capnp::schema::Type::LIST:
		case capnp::schema::Type::STRUCT:
		case capnp::schema::Type::INTERFACE:
		case capnp::schema::Type::ANY_POINTER:
		case capnp::schema::Type::ENUM:
			formatString = "O";
			elementSize = sizeof(PyObject*);
			break;
		
		case capnp::schema::Type::INT8:
			formatString="<b";
			elementSize = 1;
			break;
		case capnp::schema::Type::INT16:
			formatString="<h";
			elementSize = 2;
			break;
		case capnp::schema::Type::INT32:
			formatString="<i";
			elementSize = 4;
			break;
		case capnp::schema::Type::INT64:
			formatString="<q";
			elementSize = 8;
			break;
		
		case capnp::schema::Type::UINT8:
			formatString="<B";
			elementSize = 1;
			break;
		case capnp::schema::Type::UINT16:
			formatString="<H";
			elementSize = 2;
			break;
		case capnp::schema::Type::UINT32:
			formatString="<I";
			elementSize = 4;
			break;
		case capnp::schema::Type::UINT64:
			formatString="<Q";
			elementSize = 8;
			break;
			
		case capnp::schema::Type::FLOAT32:
			formatString="<f";
			elementSize = 4;
			break;
		case capnp::schema::Type::FLOAT64:
			formatString="<d";
			elementSize = 8;
			break;
			
		case capnp::schema::Type::BOOL:
			formatString="?";
			elementSize = 1;
			break;
	}
	
	return tuple(elementSize, formatString);
}

PyArray_Descr* numpyWireType(capnp::Type type) {
	#define HANDLE_TYPE(cap_name, npy_name) \
		case capnp::schema::Type::cap_name: { \
			PyArray_Descr* baseType = PyArray_DescrFromType(npy_name); \
			PyArray_Descr* littleEndianType = PyArray_DescrNewByteorder(baseType, NPY_LITTLE); \
			Py_DECREF(baseType); \
			return littleEndianType; \
		}
	
	#define HANDLE_OBJECT(cap_name) \
		case capnp::schema::Type::cap_name: { \
			return PyArray_DescrFromType(NPY_OBJECT); \
		}
	
	switch(type.which()) {				
		HANDLE_TYPE(INT8,  NPY_INT8);
		HANDLE_TYPE(INT16, NPY_INT16);
		HANDLE_TYPE(INT32, NPY_INT32);
		HANDLE_TYPE(INT64, NPY_INT64);
		
		HANDLE_TYPE(UINT8,  NPY_UINT8);
		HANDLE_TYPE(UINT16, NPY_UINT16);
		HANDLE_TYPE(UINT32, NPY_UINT32);
		HANDLE_TYPE(UINT64, NPY_UINT64);
		
		HANDLE_TYPE(FLOAT32, NPY_FLOAT32);
		HANDLE_TYPE(FLOAT64, NPY_FLOAT64);
		
		HANDLE_TYPE(BOOL, NPY_BOOL);
		
		HANDLE_OBJECT(VOID);
		HANDLE_OBJECT(TEXT);
		HANDLE_OBJECT(DATA);
		HANDLE_OBJECT(LIST);
		HANDLE_OBJECT(STRUCT);
		HANDLE_OBJECT(INTERFACE);
		HANDLE_OBJECT(ANY_POINTER);
		
		HANDLE_OBJECT(ENUM);
	}
	
	#undef HANDLE_TYPE
	#undef HANDLE_OBJECT
	
	KJ_UNREACHABLE;
}

struct BuilderSlot {
	mutable capnp::Type type;
	
	BuilderSlot(capnp::Type type) : type(type) {}
	
	virtual void set(capnp::DynamicValue::Reader other) const = 0;
	virtual void adopt(capnp::Orphan<DynamicValue>&& orphan) const = 0;
	virtual DynamicValue::Builder get() const = 0;
	virtual DynamicValue::Builder init(unsigned int size) const = 0;
	
	virtual ~BuilderSlot() {};
};

struct FieldSlot : public BuilderSlot {
	mutable DynamicStruct::Builder builder;
	mutable capnp::StructSchema::Field field;
	
	FieldSlot(DynamicStruct::Builder builder, capnp::StructSchema::Field field) :
		BuilderSlot(field.getType()),
		builder(builder), field(field)
	{}
	
	void set(DynamicValue::Reader newVal) const override { builder.set(field, newVal); }
	void adopt(capnp::Orphan<DynamicValue>&& orphan) const override { builder.adopt(field, mv(orphan)); }
	DynamicValue::Builder get() const override { return builder.get(field); }
	DynamicValue::Builder init(unsigned int size) const override { return builder.init(field, size); }
};

struct ListItemSlot : public BuilderSlot {
	mutable DynamicList::Builder list;
	mutable uint32_t index;
	
	ListItemSlot(DynamicList::Builder list, uint32_t index) :
		BuilderSlot(list.getSchema().getElementType()),
		list(list), index(index)
	{}
	
	void set(DynamicValue::Reader newVal) const override { list.set(index, newVal); }
	void adopt(capnp::Orphan<DynamicValue>&& orphan) const override { list.adopt(index, mv(orphan)); }
	DynamicValue::Builder get() const override { return list[index]; }
	DynamicValue::Builder init(unsigned int size) const override { return list.init(index, size); }
	
};

void assign(const BuilderSlot& dst, py::object object);

bool isTensor(capnp::Type type) {
	if(!type.isStruct())
		return false;
	
	auto asStruct = type.asStruct();
	
	KJ_IF_MAYBE(pDataField, asStruct.findFieldByName("data")) {
		KJ_IF_MAYBE(pShapeField, asStruct.findFieldByName("shape")) {
			if(!pDataField->getType().isList())
				return false;
			
			if(!pShapeField->getType().isList())
				return false;
			
			return true;
		}
	}
	
	return false;
}

py::buffer getAsBufferViaNumpy(py::object input, capnp::Type type, int minDims, int maxDims) {
	PyArray_Descr* wireType = numpyWireType(type);
	
	if(wireType == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, kj::str("Requested type has no corresponding NumPy equivalent").cStr());
		throw py::error_already_set();
	}
	
	// Steals wireType
	PyObject* arrayObject = PyArray_FromAny(input.ptr(), wireType, minDims, maxDims, NPY_ARRAY_C_CONTIGUOUS, nullptr);
	if(arrayObject == nullptr)
		throw py::error_already_set();
	
	return py::reinterpret_steal<py::buffer>(arrayObject);
}

void setObjectTensor(DynamicStruct::Builder dsb, py::buffer_info& bufinfo) {
	// Check whether array is contiguous
	size_t expectedStride = sizeof(PyObject*);
	for(int dimension = bufinfo.shape.size() - 1; dimension >= 0; --dimension) {
		KJ_REQUIRE(expectedStride == bufinfo.strides[dimension], "Array is not contiguous C-order", dimension);
		expectedStride *= bufinfo.shape[dimension];
	}
	
	// Check whether size matches expectation
	{
		size_t shapeProd = 1;
		for(uint64_t e : bufinfo.shape) shapeProd *= e;
		KJ_REQUIRE(shapeProd == bufinfo.size, "prod(shape) must equal data.size()");
	}
	
	capnp::DynamicList::Builder shape = dsb.init("shape", bufinfo.shape.size()).as<DynamicList>();
	for(size_t i = 0; i < bufinfo.shape.size(); ++i)
		shape.set(i, bufinfo.shape[i]);
	
	PyObject** bufData = reinterpret_cast<PyObject**>(bufinfo.ptr);
	capnp::DynamicList::Builder data = dsb.init("data", bufinfo.size).as<DynamicList>();
	for(size_t i = 0; i < bufinfo.size; ++i) {
		assign(ListItemSlot(data, i), py::reinterpret_borrow<py::object>(bufData[i]));
	}
}

void setBoolTensor(DynamicStruct::Builder dsb, py::buffer_info& bufinfo) {
	// Check format
	capnp::ListSchema schema = dsb.getSchema().getFieldByName("data").getType().asList();
	KJ_REQUIRE(schema.getElementType().isBool());
	
	// Check whether array is contiguous
	size_t expectedStride = 1;
	for(int dimension = bufinfo.shape.size() - 1; dimension >= 0; --dimension) {
		KJ_REQUIRE(expectedStride == bufinfo.strides[dimension], "Array is not contiguous C-order", dimension);
		expectedStride *= bufinfo.shape[dimension];
	}
	
	// Check whether size matches expectation
	{
		size_t shapeProd = 1;
		for(uint64_t e : bufinfo.shape) shapeProd *= e;
		KJ_REQUIRE(shapeProd == bufinfo.size, "prod(shape) must equal data.size()");
	}
	
	capnp::DynamicList::Builder shape = dsb.init("shape", bufinfo.shape.size()).as<DynamicList>();
	for(size_t i = 0; i < bufinfo.shape.size(); ++i)
		shape.set(i, bufinfo.shape[i]);
	
	unsigned char* bufData = reinterpret_cast<unsigned char*>(bufinfo.ptr);
	capnp::DynamicList::Builder data = dsb.init("data", bufinfo.size).as<DynamicList>();
	for(size_t i = 0; i < bufinfo.size; ++i)
		data.set(i, bufData[i] != 0);
}

void setDataTensor(DynamicStruct::Builder dsb, py::buffer_info& bufinfo) {
	// Check format
	capnp::ListSchema schema = dsb.getSchema().getFieldByName("data").getType().asList();
	
	auto format = pyFormat(schema.getElementType());
	size_t elementSize = kj::get<0>(format);
	kj::StringPtr expectedFormat = kj::get<1>(format);
	
	kj::String actualFormat = kj::str(bufinfo.format.c_str());
	
	if(actualFormat[0] == '!')
		actualFormat[0] = '>';
	
	// If we get a default ordering, check if we are on little endian CPU
	if(actualFormat[0] != '<' && actualFormat[0] != '>') {
		#if __BYTE_ORDERING__ == __LITTLE_ENDIAN__
			actualFormat = str("<", actualFormat);
		#else
			actualFormat = str(">", actualFormat);
		#endif
	}
	
	KJ_REQUIRE(actualFormat == expectedFormat, "Can only assign data of compatible data types", expectedFormat, actualFormat);
	KJ_REQUIRE(bufinfo.itemsize == elementSize, "Apparently python and I have different ideas about the size of this type");
	
	// Check whether array is contiguous
	size_t expectedStride = elementSize;
	for(int dimension = bufinfo.shape.size() - 1; dimension >= 0; --dimension) {
		KJ_REQUIRE(expectedStride == bufinfo.strides[dimension], "Array is not contiguous C-order", dimension);
		expectedStride *= bufinfo.shape[dimension];
	}
	
	// Check whether size matches expectation
	{
		size_t shapeProd = 1;
		for(uint64_t e : bufinfo.shape) shapeProd *= e;
		KJ_REQUIRE(shapeProd == bufinfo.size, "prod(shape) must equal data.size()");
	}
	
	capnp::DynamicList::Builder shape = dsb.init("shape", bufinfo.shape.size()).as<DynamicList>();
	for(size_t i = 0; i < bufinfo.shape.size(); ++i)
		shape.set(i, bufinfo.shape[i]);
	
	capnp::DynamicList::Builder data = dsb.init("data", bufinfo.size).as<DynamicList>();
	byte* dataPtr = const_cast<byte*>(((capnp::AnyList::Builder) data).asReader().getRawBytes().begin());
	memcpy(dataPtr, bufinfo.ptr, elementSize * bufinfo.size);
}

//! Allows assigning tensor types from a buffer
void setTensor(DynamicStruct::Builder dsb, py::buffer buffer) {	
	py::buffer_info bufinfo = buffer.request();
	
	// Check format
	capnp::ListSchema schema = dsb.getSchema().getFieldByName("data").getType().asList();
	
	if(isObjectType(schema.getElementType())) {
		setObjectTensor(dsb, bufinfo);
		return;
	}
	
	if(schema.getElementType().isBool()) {
		setBoolTensor(dsb, bufinfo);
		return;
	}
	
	setDataTensor(dsb, bufinfo);
}

void setTensor(DynamicValue::Builder dvb, py::buffer buffer) {
	//TODO: Derive tensor type from buffer value?
	setTensor(dvb.as<DynamicStruct>(), buffer);
}

void assign(const BuilderSlot& dst, py::object object) {
	auto assignmentFailureLog = kj::strTree();
	
	// Attempt 1: Check if target is orphan that can be adopted
	pybind11::detail::make_caster<capnp::Orphan<DynamicValue>> orphanCaster;
	if(orphanCaster.load(object, false)) {
		capnp::Orphan<DynamicValue>& orphanRef = (capnp::Orphan<DynamicValue>&) orphanCaster;
		dst.adopt(mv(orphanRef));
		return;
	}
	
	// Attempt 2: Check if target can be converted into a reader directly
	pybind11::detail::make_caster<DynamicValue::Reader> dynValCaster;
	if(dynValCaster.load(object, false)) {
		dst.set((DynamicValue::Reader&) dynValCaster);
		return;
	}
	
	// Attempt 3: Try to assign from a sequence
	if(py::sequence::check_(object) && dst.type.isList()) {
		auto asSequence = py::reinterpret_borrow<py::sequence>(object);
		
		DynamicList::Builder listDst = dst.init(asSequence.size()).as<DynamicList>();
		for(unsigned int i = 0; i < listDst.size(); ++i) {
			assign(ListItemSlot(listDst, i), asSequence[i]);
		}
		
		return;
	} else {
		if(!dst.type.isList()) {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from sequence because slot type is not list\n");
		}
	}
	
	// Attempt 4: If we are a tensor, try to convert via a numpy array
	if(isTensor(dst.type)) {
		auto scalarType = dst.type.asStruct().getFieldByName("data").getType().asList().getElementType();
		
		py::buffer targetBuffer;
		try {
			targetBuffer = getAsBufferViaNumpy(object, scalarType, 0, 100);
		} catch(py::error_already_set& e) {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Could not obtain buffer from numpy due to following reason: ", e.what(), "\n");
			goto tensor_conversion_failed;
		}
		
		// From now on, we don't wanna catch exceptions, as this should
		// always work
		setTensor(dst.get(), targetBuffer);
		return;
	} else {
		assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from array because slot type is not a tensor\n");
	}
	
	// This label exists in case we add more conversion routines later
	tensor_conversion_failed:
	
	throw std::invalid_argument(kj::str("Could not find a way to assign object of type ", py::cast<kj::StringPtr>(py::str(py::type::of(object))), ".\n", assignmentFailureLog.flatten()).cStr());
}

void setItem(capnp::DynamicList::Builder list, unsigned int index, py::object value) {
	assign(ListItemSlot(list.as<DynamicList>(), index), mv(value));
}

void setFieldByName(capnp::DynamicStruct::Builder builder, kj::StringPtr fieldName, py::object value) {
	assign(FieldSlot(builder, builder.getSchema().getFieldByName(fieldName)), mv(value));
}

void setField(capnp::DynamicStruct::Builder builder, capnp::StructSchema::Field field, py::object value) {
	assign(FieldSlot(builder, field), mv(value));
}

DynamicValue::Builder initField(capnp::DynamicStruct::Builder builder, capnp::StructSchema::Field field) {
	return builder.init(field);
}

DynamicValue::Builder initFieldByName(capnp::DynamicStruct::Builder, kj::StringPtr fieldName) {
	return builder.init(fieldName);
}

DynamicValue::Builder initList(capnp::DynamicStruct::Builder builder, capnp::StructSchema::Field field, size_t size) {
	return builder.init(field, size);
}

DynamicValue::Builder initListByName(capnp::DynamicStruct::Builder builder, kj::StringPtr listName, size_t size) {
	return builder.init(listName, size);
}

capnp::AnyList::Reader asAnyListReader(capnp::DynamicList::Reader reader) {
	return reader;
}

capnp::AnyList::Reader asAnyListReader(capnp::DynamicList::Builder builder) {
	return asAnyListReader(builder.asReader());
}

template<typename T>
py::buffer_info getDataTensor(T& tensor, bool readOnly) {
	try {
		// Extract shape and dat
		auto shape = tensor.get("shape").template as<capnp::List<uint64_t>>();
		auto data  = tensor.get("data").template as<capnp::DynamicList>();
				
		// Extract raw data
		kj::ArrayPtr<const byte> rawBytes = asAnyListReader(data).getRawBytes();
		byte* bytesPtr = const_cast<byte*>(rawBytes.begin());
				
		// Extract format
		capnp::ListSchema schema = data.getSchema();
		auto format = pyFormat(schema.getElementType());
		size_t elementSize = kj::get<0>(format);
		kj::StringPtr formatString = kj::get<1>(format);
		
		// Sanity checks
		KJ_REQUIRE(elementSize * data.size() == rawBytes.size());
		
		{
			size_t shapeProd = 1;
			for(uint64_t e : shape) shapeProd *= e;
			KJ_REQUIRE(shapeProd == data.size(), "prod(shape) must equal data.size()");
		}
		
		std::vector<size_t> outShape(shape.size());
		std::vector<size_t> strides(shape.size());
		
		size_t stride = elementSize;
		for(int i = shape.size() - 1; i >= 0; --i) {
			outShape[i] = shape[i];
			strides[i] = stride;
			stride *= shape[i];
		}
		
		return py::buffer_info(
			(void*) bytesPtr, elementSize, std::string(formatString.cStr()), shape.size(), outShape, strides, readOnly
		);
	} catch(kj::Exception& error) {
		KJ_LOG(ERROR, "Failed to create python buffer. See below error\n", error);
	} 
	
	return py::buffer_info((byte*) nullptr, 0);
}

template<typename T>
py::buffer_info getBoolTensor(T& tensor) {
	// Extract shape and dat
	auto shape = tensor.get("shape").template as<capnp::List<uint64_t>>();
	auto data  = tensor.get("data").template as<capnp::DynamicList>();
	
	auto resultHolder = new ContiguousCArray();
	
	auto outData = resultHolder -> alloc<uint8_t>(shape);
	resultHolder -> format = kj::str("?");
	
	for(auto i : kj::indices(data)) {
		outData[i] = data[i].template as<bool>() ? 1 : 0;
	}
	
	py::buffer asPyBuffer = py::cast(resultHolder);
	return asPyBuffer.request(true);
}

template<typename T>
py::buffer_info getObjectTensor(py::object pySelf, T& tensor) {
	// Extract shape and dat
	auto shape = tensor.get("shape").template as<capnp::List<uint64_t>>();
	auto data  = tensor.get("data").template as<capnp::DynamicList>();
	
	auto resultHolder = new ContiguousCArray();
	
	auto outData = resultHolder -> alloc<PyObject*>(shape);
	resultHolder -> format = kj::str("O");
	
	for(auto i : kj::indices(data)) {
		py::object outObject = py::cast(data[i]);
		
		if(needsBackReference(data[i]))
			outObject.attr("_parent") = pySelf;
		
		outData[i] = outObject.inc_ref().ptr();
	}
	
	py::buffer asPyBuffer = py::cast(resultHolder);
	return asPyBuffer.request(true);
}

template<typename T>
py::buffer_info getTensor(py::object self, bool readOnly) {
	T tensor = py::cast<T>(self);
	
	auto schema = tensor.getSchema();
	auto scalarType = schema.getFieldByName("data").getType().asList().getElementType();
	
	if(isObjectType(scalarType)) {
		return getObjectTensor(self, tensor);
	}
	
	if(scalarType.isBool()) {
		return getBoolTensor(tensor);
	}
	
	return getDataTensor(tensor, readOnly);
}

//! Defines the buffer protocol for a type T which must be a capnp::List<...>::{Builder, Reader}
template<typename T>
void defListBuffer(py::class_<T>& c, bool readOnly) {
	c.def_buffer([readOnly](T& list) {	
		try {	
			// Extract raw data
			kj::ArrayPtr<const byte> rawBytes = asAnyListReader(list).getRawBytes();
			byte* bytesPtr = const_cast<byte*>(rawBytes.begin());
			
			// Read format
			capnp::ListSchema schema = list.getSchema();
			auto format = pyFormat(schema.getElementType());
			size_t elementSize = kj::get<0>(format);
			kj::StringPtr formatString = kj::get<1>(format);
			
			// Sanity checks
			KJ_REQUIRE(elementSize * list.size() == rawBytes.size());
					
			return py::buffer_info(
				(void*) bytesPtr, elementSize, std::string(formatString.cStr()), list.size(), readOnly
			);
		} catch(kj::Exception& error) {
			KJ_LOG(ERROR, "Failed to create python buffer. See below error\n", error);
		}
		
		return py::buffer_info((byte*) nullptr, 0);
	});
}

//! Defines the buffer protocol for a capnp::DynamicStruct::{Reader, Builder} that has a "shape" and a "data" field
template<typename T>
void defTensorBuffer(py::class_<T>& c, bool readOnly) {
	c.def_buffer([readOnly](py::object tensor) {
		return getTensor<T>(tensor, readOnly);
	});
}

//! Binds capnp::DynamicList::{Builder, Reader}
void bindListClasses(py::module_& m) {
	using DLB = DynamicList::Builder;
	using DLR = DynamicList::Reader;
	
	py::class_<DLB> cDLB(m, "DynamicListBuilder", py::dynamic_attr(), py::buffer_protocol());
	defGetItemAndLen(cDLB);
	cDLB.def("__setitem__", &setItem);
	defListBuffer(cDLB, false);
	
	py::class_<DLR> cDLR(m, "DynamicListReader", py::dynamic_attr(), py::buffer_protocol());
	defGetItemAndLen(cDLR);
	defListBuffer(cDLR, true);
}

//! Defines the _get methods needed by all accessors, as well as the dynamic "get" and "__getitem__" methods
template<typename T, typename... Extra>
void defGet(py::class_<T, Extra...>& c) {
	c.def("_get", [](T& self, capnp::StructSchema::Field field) { return underscoreGet<T, kj::StringPtr>(self, field.getProto().getName()); });
	c.def("_get", &underscoreGet<T, kj::StringPtr>);
	
	auto genericGet = [](py::object ds, py::object field) { return getField(ds, field); };
	
	c.def("get", genericGet);
	c.def("__getitem__", genericGet);
	
	// Register class as ABC
	auto abcModule = py::module_::import("collections.abc");
	abcModule.attr("Mapping").attr("register")(c);
}

template<typename T, typename... Extra>
void defHas(py::class_<T, Extra...>& c) {
	c.def("has", [](T& ds, kj::StringPtr name) { return ds.has(name); });
}

template<typename T, typename... Extra>
void defWhich(py::class_<T, Extra...>& c) {
	c.def("which", [](T& ds) -> py::object {
		auto maybeField = ds.which();
		
		KJ_IF_MAYBE(pField, maybeField) {
			return py::cast(pField->getProto().getName());
		}
		
		return py::none();
	});
}

static uint64_t graphNodeUUID = 0;

bool canAddToGraph(capnp::Type type) {
	if(type.isStruct()) return true;
	if(type.isInterface()) return true;
	
	if(type.isList()) {
		auto wrapped = type.asList().getElementType();
		
		if(wrapped.isStruct()) return true;
		if(wrapped.isInterface()) return true;
		if(wrapped.isList()) return true;
	}
	
	return false;
}

kj::Vector<uint64_t> addDynamicToGraph(py::object graph, capnp::DynamicValue::Reader reader);

void addGraphNode(py::object graph, uint64_t id, kj::StringPtr label) {
	graph.attr("node")(kj::str(id), label);
}
void addGraphEdge(py::object graph, uint64_t id1, uint64_t id2, kj::StringPtr label) {
	graph.attr("edge")(kj::str(id1), kj::str(id2), label);
}

uint64_t addStructToGraph(py::object graph, capnp::DynamicStruct::Reader reader) {
	auto schema = reader.getSchema();
	
	auto nodeCaption = kj::strTree(schema.getUnqualifiedName());
	
	uint64_t nodeId = graphNodeUUID++;
	
	kj::Function<void(capnp::StructSchema::Field, capnp::DynamicStruct::Reader)> handleField;
	kj::Function<void(capnp::DynamicStruct::Reader)> handleGroup;
	
	handleField = [&](capnp::StructSchema::Field field, capnp::DynamicStruct::Reader reader) {
		auto type = field.getType();
		
		if(canAddToGraph(type)) {
			auto children = addDynamicToGraph(graph, reader.get(field));
			for(uint64_t child : children) {
				addGraphEdge(graph, nodeId, child, field.getProto().getName());
			}
		} else {
			nodeCaption = kj::strTree(mv(nodeCaption), "\n", field.getProto().getName(), " = ", reader.get(field));
		}
	};
	
	handleGroup = [&](capnp::DynamicStruct::Reader reader) {
		auto schema = reader.getSchema();
		
		uint64_t nSubGroups = 0;
		
		KJ_IF_MAYBE(pField, reader.which()) {
			if(pField -> getProto().isGroup())
				++nSubGroups;
		}
				
		for(auto field : schema.getNonUnionFields()) {
			if(field.getProto().isGroup())
				++nSubGroups;
		}
		
		KJ_IF_MAYBE(pField, reader.which()) {
			auto& field = *pField;
			if(field.getProto().isGroup() && nSubGroups <= 1) {
			} else {
				nodeCaption = kj::strTree(mv(nodeCaption), "\n", field.getProto().getName());
				handleField(field, reader);
			}
		}
				
		for(auto field : schema.getNonUnionFields()) {
			if(field.getProto().isGroup() && nSubGroups <= 1) {
			} else {
				handleField(field, reader);
			}
		}
		
		KJ_IF_MAYBE(pField, reader.which()) {
			auto& field = *pField;
			if(field.getProto().isGroup() && nSubGroups <= 1) {
				nodeCaption = kj::strTree(mv(nodeCaption), "\n", "-- ", field.getProto().getName(), " --");
				handleGroup(reader.get(field).as<capnp::DynamicStruct>());
			} else {
			}
		}
				
		for(auto field : schema.getNonUnionFields()) {
			if(field.getProto().isGroup() && nSubGroups <= 1) {
				nodeCaption = kj::strTree(mv(nodeCaption), "\n", "-- ", field.getProto().getName(), " --");
				handleGroup(reader.get(field).as<capnp::DynamicStruct>());
			} else {
			}
		}
	};
	
	handleGroup(reader);
	
	addGraphNode(graph, nodeId, nodeCaption.flatten());
	return nodeId;
}

uint64_t addInterfaceToGraph(py::object graph, capnp::DynamicCapability::Client client) {
	auto schema = client.getSchema();
	
	uint64_t nodeId = graphNodeUUID++;
	addGraphNode(graph, nodeId, kj::str(schema));
	
	return nodeId;
}
	
kj::Vector<uint64_t> addDynamicToGraph(py::object graph, capnp::DynamicValue::Reader reader) {
	auto type = reader.getType();
		
	kj::Vector<uint64_t> result;
	if(type == capnp::DynamicValue::STRUCT) {
		result.add(addStructToGraph(graph, reader.as<capnp::DynamicStruct>()));
		
	} else if(type == capnp::DynamicValue::CAPABILITY) {
		result.add(addInterfaceToGraph(graph, reader.as<capnp::DynamicCapability>()));
		
	} else if(type == capnp::DynamicValue::LIST) {
		auto asList = reader.as<capnp::DynamicList>();
		auto wrapped = asList.getSchema().getElementType();
		
		for(auto entry : asList) {
			if(wrapped.isList()) {
				if(canAddToGraph(wrapped.asList().getElementType())) {
					uint64_t nodeId = graphNodeUUID++;
					addGraphNode(graph, nodeId, "List");
					
					for(uint64_t child : addDynamicToGraph(graph, entry)) {
						addGraphEdge(graph, nodeId, child, "");
					}
					
					result.add(nodeId);
				} else {
					uint64_t nodeId = graphNodeUUID++;
					addGraphNode(graph, nodeId, kj::str(
						"List\n",
						entry
					));
					result.add(nodeId);
				}
			} else if(wrapped.isStruct()) {
				result.add(addStructToGraph(graph, entry.as<capnp::DynamicStruct>()));
			} else if(wrapped.isInterface()) {
				result.add(addInterfaceToGraph(graph, entry.as<capnp::DynamicCapability>()));
			} else {
				KJ_FAIL_REQUIRE("Added un-renderable list type to graph");
			}
		}
	} else {
		KJ_FAIL_REQUIRE("Added un-renderable list type to graph", type);
	}
	
	return result;
}

py::object visualize(capnp::DynamicStruct::Reader reader, py::kwargs kwargs) {
	py::object graph = py::module_::import("graphviz").attr("Digraph")(**kwargs);
	addStructToGraph(graph, reader);
	
	return graph;
}

void bindStructClasses(py::module_& m) {
	using DSB = DynamicStruct::Builder;
	using DSR = DynamicStruct::Reader;
	using DSP = DynamicStructPipeline;
	using DST = fsc::Temporary<DynamicStruct>;
	
	// ----------------- BUILDER ------------------
	
	py::class_<DSB> cDSB(m, "DynamicStructBuilder", py::dynamic_attr(), py::multiple_inheritance(), py::metaclass(*baseMetaType), py::buffer_protocol());
	cDSB.def(py::init([](DynamicStruct::Builder b) { return b; }));
	
	defGet(cDSB);
	defHas(cDSB);
	defWhich(cDSB);
	
	defTensorBuffer(cDSB, false);
	
	cDSB.def("set", &setField);
	cDSB.def("set", &setFieldByName);
	
	cDSB.def("init", &initField);
	cDSB.def("init", &initFieldByName);
	cDSB.def("init", &initList);
	cDSB.def("init", &initListByName);
		
	cDSB.def("__setitem__", &setField);
	cDSB.def("__setitem__", &setFieldByName);
		
	cDSB.def("__delitem__", [](DSB& dsb, kj::StringPtr name) { dsb.clear(name); });
	
	cDSB.def("__len__", [](DSB& ds) {
		auto schema = ds.getSchema();
		size_t result = schema.getNonUnionFields().size();
		if(schema.getUnionFields().size() > 0)
			result += 1;
		
		return result;
	});
	
	cDSB.def("items", [](DSB& reader) {
		py::list result;
		
		auto schema = reader.getSchema();
		
		for(auto field : schema.getNonUnionFields())
			result.append(py::make_tuple(field.getProto().getName(), reader.get(field)));
		
		KJ_IF_MAYBE(pField, reader.which()) {
			result.append(py::make_tuple(pField->getProto().getName(), reader.get(*pField)));
		}
		
		return result;
	});
	
	cDSB.def("__iter__", [](DSB& ds) {
		py::list result;
		
		KJ_IF_MAYBE(pField, ds.which()) {
			result.append(str(pField -> getProto().getName()));
		}
		
		for(auto field : ds.getSchema().getNonUnionFields()) {
			result.append(str(field.getProto().getName()));
		}
		
		return py::eval("iter")(result);
	});
	
	cDSB.def("disown", [](py::object self, kj::StringPtr field) {
		DSB builder = py::cast<DSB>(self);
		py::object result = py::cast(builder.disown(field));
		result.attr("_src") = self;
		return result;
	});
	
	cDSB.def("clone", [](DSB& self) {
		auto msg = new capnp::MallocMessageBuilder();
		msg->setRoot(self.asReader());
		
		auto pyBuilder = py::cast(capnp::DynamicValue::Builder(msg->getRoot<capnp::DynamicStruct>(self.getSchema())));
		
		auto pyMsg = py::cast(msg);
		pyBuilder.attr("_msg") = pyMsg;
		
		return pyBuilder;
	});
	
	cDSB.def("__repr__", [](DSB& self) {
		return kj::str(self);
	});
	
	cDSB.def("pretty", [](DSB& self) {
		return capnp::prettyPrint(self).flatten();
	});
	
	m.def("totalSize", [](DSB& builder) { return builder.totalSize().wordCount * 8; });
	
	m.def("visualize", &visualize);
	m.def("visualize", [](capnp::DynamicStruct::Builder b, py::kwargs kwargs) { return visualize(b.asReader(), kwargs); });
	
	// ----------------- READER ------------------
	
	py::class_<DSR> cDSR(m, "DynamicStructReader", py::dynamic_attr(), py::multiple_inheritance(), py::metaclass(*baseMetaType), py::buffer_protocol());
	cDSR.def(py::init([](DynamicStruct::Reader r) { return r; }));
	
	defGet(cDSR);
	defHas(cDSR);
	defWhich(cDSR);
	
	defTensorBuffer(cDSR, true);
	
	cDSR.def("__len__", [](DSR& ds) {
		auto schema = ds.getSchema();
		size_t result = schema.getNonUnionFields().size();
		if(schema.getUnionFields().size() > 0)
			result += 1;
		
		return result;
	});
	
	cDSR.def("__iter__", [](DSR& ds) {
		py::list result;
		
		KJ_IF_MAYBE(pField, ds.which()) {
			result.append(str(pField -> getProto().getName()));
		}
		
		for(auto field : ds.getSchema().getNonUnionFields()) {
			result.append(str(field.getProto().getName()));
		}
		
		return py::eval("iter")(result);
	});
	
	cDSR.def("items", [](DSR& reader) {
		py::list result;
		
		auto schema = reader.getSchema();
		
		for(auto field : schema.getNonUnionFields())
			result.append(py::make_tuple(field.getProto().getName(), reader.get(field)));
		
		KJ_IF_MAYBE(pField, reader.which()) {
			result.append(py::make_tuple(pField->getProto().getName(), reader.get(*pField)));
		}
		
		return result;
	});
	
	cDSR.def("clone", [](DSR& self) {
		auto msg = new capnp::MallocMessageBuilder();
		msg->setRoot(self);
		
		auto pyBuilder = py::cast(capnp::DynamicValue::Builder(msg->getRoot<capnp::DynamicStruct>(self.getSchema())));
		
		auto pyMsg = py::cast(msg);
		pyBuilder.attr("_msg") = pyMsg;
		
		return pyBuilder;
	});
	
	cDSR.def("__repr__", [](DSR& self) {
		return kj::str(self);
	});
	
	cDSR.def("pretty", [](DSR& self) {
		return capnp::prettyPrint(self).flatten();
	});
	
	m.def("totalSize", [](DSR& reader) { return reader.totalSize().wordCount * 8; });
	
	// ----------------- PIPELINE ------------------
	
	py::class_<DSP> cDSP(m, "DynamicStructPipeline", py::multiple_inheritance(), py::metaclass(*baseMetaType));
	defGet(cDSP);
	
	// TODO: This is a little bit hacky, but currently the only way we can allow the construction of derived instances
	cDSP.def(py::init([](DynamicStructPipeline& p) {
		//KJ_REQUIRE((key == INTERNAL_ACCESS_KEY), "The pipeline constructor is reserved for internal use");
		return p;
	}));
	
	cDSP.def("__len__", [](DSP& ds) {
		size_t result = ds.schema.getNonUnionFields().size();
		
		return result;
	});
	
	cDSP.def("__iter__", [](DSP& ds) {
		py::list result;
		
		for(auto field : ds.schema.getNonUnionFields()) {
			result.append(str(field.getProto().getName()));
		}
		
		return py::eval("iter")(result);
	});
	
	cDSP.def("items", [](DSP& reader) {
		py::list result;
		
		auto schema = reader.getSchema();
		
		for(auto field : schema.getNonUnionFields())
			result.append(py::make_tuple(field.getProto().getName(), reader.get(field)));
		
		return result;
	});
	
	
	// ----------------- OTHERS ------------------
	
	py::class_<capnp::Response<AnyPointer>>(m, "DynamicResponse");
	
	py::class_<capnp::Orphan<DynamicValue>>(m, "DynamicOrphan", py::dynamic_attr())
		.def_property_readonly("val", [](capnp::Orphan<DynamicValue>& self) { return self.get(); }, py::keep_alive<0, 1>())
	;
	
	py::class_<DST, DSB> cDST(m, "DynamicMessage", py::metaclass(*baseMetaType));
	
	// Conversion
	
	py::implicitly_convertible<DSB, DSR>();
}

void bindFieldDescriptors(py::module_& m) {
	py::class_<capnp::StructSchema::Field>(m, "Field")		
		.def("__get__", [](capnp::StructSchema::Field& field, py::object self, py::object type) -> py::object {
			if(self.is_none())
				return py::cast(kj::str("Field ", field.getProto().getName(), " : ", typeName(field.getType())));
			
			return getField(self, py::cast(field));
		})
		
		// Note the argument reversal here
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, py::object value) { setField(self, field, value); })
		
		.def("__delete__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self) { self.clear(field); })
	;
}

void bindMessageBuilders(py::module_& m) {
	py::class_<capnp::MessageBuilder>(m, "MessageBuilder")
		.def("setRoot", [](py::object self, DynamicStruct::Builder newRoot) {
			capnp::MessageBuilder& builder = py::cast<capnp::MessageBuilder&>(self);
			builder.setRoot(newRoot.asReader());
			py::object result = py::cast(builder.getRoot<DynamicStruct>(newRoot.getSchema()));
			result.attr("_msg") = self;
			return result;
		})
		.def("setRoot", [](py::object self, capnp::Orphan<DynamicValue>& newRoot) {
			auto type = newRoot.getType();
			KJ_REQUIRE(type == DynamicValue::STRUCT, "Can only adopt struct orphans", type);
			
			auto structRoot = newRoot.releaseAs<DynamicStruct>();
			auto schema = structRoot.get().getSchema();
			
			capnp::MessageBuilder& builder = py::cast<capnp::MessageBuilder&>(self);
			builder.adoptRoot(mv(structRoot));
			
			py::object result = py::cast(builder.getRoot<DynamicStruct>(schema));
			result.attr("_msg") = self;
			return result;
		})
		.def("initRoot", [](py::object self, py::object type) {
			return type.attr("_initRootAs")(self);
		})
	;
	py::class_<capnp::MallocMessageBuilder, capnp::MessageBuilder>(m, "MallocMessageBuilder")
		.def(py::init())
	;
}

void bindCapClasses(py::module_& m) {
	py::class_<capnp::DynamicCapability::Client>(m, "DynamicCapabilityClient", py::multiple_inheritance(), py::metaclass(*baseMetaType))
		.def(py::init([](capnp::DynamicCapability::Client src) { return src; }))
	;
	py::class_<capnp::DynamicCapability::Server>(m, "DynamicCapabilityServer", py::multiple_inheritance(), py::metaclass(*baseMetaType));
}

void bindEnumClasses(py::module_& m) {
	py::class_<DynamicEnum>(m, "DynamicEnum")
		.def("__repr__", [](DynamicEnum& self) {
			KJ_IF_MAYBE(pEnumerant, self.getEnumerant()) {
				return kj::str("'", pEnumerant -> getProto().getName(), "'");
			}
			return kj::str("<unknown> (", self.getRaw(), ")");
		})
		.def("__eq__", [](DynamicEnum& self, DynamicEnum& other) {
			return self.getSchema() == other.getSchema() && self.getRaw() == other.getRaw();
		}, py::is_operator())
		.def("__eq__", [](DynamicEnum& self, uint16_t other) {
			return self.getRaw() == other;
		}, py::is_operator())
		.def("__eq__", [](DynamicEnum& self, kj::StringPtr other) {
			KJ_IF_MAYBE(pEnumerant, self.getEnumerant()) {
				return pEnumerant -> getProto().getName() == other;
			}
			return false;
		}, py::is_operator())
	;
}

void bindSchemaClasses(py::module_& m) {
	py::class_<capnp::Schema>(m, "Schema");
	py::class_<capnp::StructSchema, capnp::Schema>(m, "StructSchema");
	py::class_<capnp::InterfaceSchema, capnp::Schema>(m, "InterfaceSchema");
}

uint64_t totalSize(capnp::DynamicStruct::Reader reader) {
	return reader.totalSize().wordCount * 8;
}

void bindHelpers(py::module_& m) {
	m.def("totalSize", &totalSize);
};

}

namespace fscpy {
	
	
Maybe<DynamicValue::Reader> dynamicValueFromScalar(py::handle handle) {
	// 0D arrays
	if(PyArray_IsZeroDim(handle.ptr())) {
		PyArrayObject* scalarPtr = reinterpret_cast<PyArrayObject*>(handle.ptr());
		
		switch(PyArray_TYPE(scalarPtr)) {
			#define HANDLE_NPY_TYPE(npytype, ctype) \
				case npytype: { \
					ctype* data = static_cast<ctype*>(PyArray_DATA(scalarPtr)); \
					return DynamicValue::Reader(*data); \
				}
						
			HANDLE_NPY_TYPE(NPY_INT8,  int8_t);
			HANDLE_NPY_TYPE(NPY_INT16, int16_t);
			HANDLE_NPY_TYPE(NPY_INT32, int32_t);
			HANDLE_NPY_TYPE(NPY_INT64, int64_t);
			
			HANDLE_NPY_TYPE(NPY_UINT8,  uint8_t);
			HANDLE_NPY_TYPE(NPY_UINT16, uint16_t);
			HANDLE_NPY_TYPE(NPY_UINT32, uint32_t);
			HANDLE_NPY_TYPE(NPY_UINT64, uint64_t);
			
			HANDLE_NPY_TYPE(NPY_FLOAT32, float);
			HANDLE_NPY_TYPE(NPY_FLOAT64, double);
			
			#undef HANDLE_NPY_TYPE
			
			case NPY_BOOL: {
				unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(scalarPtr)); 
				return DynamicValue::Reader((*data) != 0);
			}
				
			default:
				break;
		}
	}
	
	// NumPy scalars
	if(PyArray_IsScalar(handle.ptr(), Bool)) { \
		return DynamicValue::Reader(PyArrayScalar_VAL(handle.ptr(), Bool) != 0); \
	}
	
	#define HANDLE_TYPE(cls) \
		if(PyArray_IsScalar(handle.ptr(), cls)) { \
			return DynamicValue::Reader(PyArrayScalar_VAL(handle.ptr(), cls)); \
		}
	
	HANDLE_TYPE(Byte);
	HANDLE_TYPE(Short);
	HANDLE_TYPE(Int);
	HANDLE_TYPE(Long);
	HANDLE_TYPE(LongLong);
	
	HANDLE_TYPE(UByte);
	HANDLE_TYPE(UShort);
	HANDLE_TYPE(UInt);
	HANDLE_TYPE(ULong);
	HANDLE_TYPE(ULongLong);
	
	HANDLE_TYPE(Float);
	HANDLE_TYPE(Double);
	
	#undef HANDLE_TYPE		
	
	// Python builtins
	#define HANDLE_TYPE(ctype, pytype) \
		if(py::isinstance<pytype>(handle)) { \
			pytype typed = py::reinterpret_borrow<pytype>(handle); \
			ctype cTyped = static_cast<ctype>(typed); \
			return DynamicValue::Reader(cTyped); \
		}
		
	// Bool is a subtype of int, so this has to go first
	HANDLE_TYPE(bool, py::bool_);
	HANDLE_TYPE(signed long long, py::int_);
	HANDLE_TYPE(double, py::float_);
	
	#undef HANDLE_TYPE
	
	return nullptr;
}
	
// init method

void initCapnp(py::module_& m) {
	// Make sure numpy is initialized
	// import_array can be a macro that contains returns
	// Wrap it in a lambda
	auto importer = []() -> void* {
		import_array();
		return nullptr;
	};
	
	importer();
	if(PyErr_Occurred()) {
		throw py::error_already_set();
	}
	
	py::module_ mcapnp = m.def_submodule("capnp", "Python bindings for Cap'n'proto classes (excluding KJ library)");
	
	bindListClasses(mcapnp);
	bindBlobClasses(mcapnp);
	bindStructClasses(mcapnp);
	bindFieldDescriptors(mcapnp);
	bindMessageBuilders(mcapnp);
	bindCapClasses(mcapnp);
	bindEnumClasses(mcapnp);
	bindSchemaClasses(mcapnp);
	bindHelpers(mcapnp);
	
	m.add_object("void", py::cast(capnp::DynamicValue::Reader(capnp::Void())));
}
	
// class DynamicValuePipeline

DynamicStructPipeline DynamicValuePipeline::asStruct() {
	return DynamicStructPipeline(
		typeless.noop(), schema.asStruct()
	);
}

capnp::DynamicCapability::Client DynamicValuePipeline::asCapability() {
	capnp::Capability::Client anyCap = typeless;
	return anyCap.castAs<DynamicCapability>(schema.asInterface());
}

// class DynamicStructPipeline

DynamicValuePipeline DynamicStructPipeline::get(capnp::StructSchema::Field field) {
	capnp::AnyPointer::Pipeline typelessValue(nullptr);
	
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
	
	KJ_REQUIRE(fieldType.isStruct() || fieldType.isInterface());
	
	if(fieldType.isStruct()) {
		return DynamicValuePipeline(
			mv(typelessValue), field.getType().asStruct()
		);
	} else {
		return DynamicValuePipeline(
			mv(typelessValue), field.getType().asInterface()
		);
	}	
}

DynamicValuePipeline DynamicStructPipeline::get(kj::StringPtr name) {
	KJ_IF_MAYBE(pField, schema.findFieldByName(name)) {
		return get(*pField);
	} else {
		KJ_FAIL_REQUIRE("Field not found", name);
	}
}

}