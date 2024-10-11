#include "tensor.h"

#include "assign.h"

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;
using capnp::AnyList;

namespace py = pybind11;

namespace fscpy {

namespace {

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

py::buffer_info getBoolTensor(DynamicStruct::Reader tensor) {
	// Extract shape and dat
	auto shape = tensor.get("shape").as<capnp::List<uint64_t>>();
	auto data  = tensor.get("data").as<capnp::DynamicList>();
	
	auto resultHolder = ContiguousCArray::alloc<uint8_t>(shape, "?");
	auto outData = resultHolder.as<uint8_t>();
	
	for(auto i : kj::indices(data)) {
		outData[i] = data[i].as<bool>() ? 1 : 0;
	}
	
	py::buffer asPyBuffer = py::cast(mv(resultHolder));
	return asPyBuffer.request(true);
}

template<typename T>
py::buffer_info getObjectTensor(T tensor) {
	// Extract shape and dat
	auto shape = tensor.get("shape").asList().template as<capnp::List<uint64_t>>();
	auto data  = tensor.get("data").asList();
	
	auto resultHolder = ContiguousCArray::alloc<PyObject*>(shape, "O");
	auto outData = resultHolder.template as<PyObject*>();
	
	for(auto i : kj::indices(data)) {
		py::object outObject = py::cast(data.get(i));		
		outData[i] = outObject.inc_ref().ptr();
	}
	
	py::buffer asPyBuffer = py::cast(mv(resultHolder));
	return asPyBuffer.request(true);
}

template<typename T>
py::buffer_info getEnumTensor(T tensor) {
	// Optimized accessor for enum tensor loading
	// Pre-allocates enum values and only increases refcounts
	// Therefore bypasses lots of object allocation and casting logic	
	
	// Extract shape and data
	auto shape = tensor.get("shape").asList().template as<capnp::List<uint64_t>>();
	auto data  = tensor.get("data").asList();
	
	// Pre-allocate enums
	auto enumSchema = data.getSchema().getElementType().asEnum();
	auto enumerants = enumSchema.getEnumerants();
	auto knownEnumerants = [&]() {
		auto builder = kj::heapArrayBuilder<py::object>(enumerants.size());
		for(auto& e : enumerants) builder.add(py::cast(EnumInterface(e)));
		return builder.finish();
	}();
	
	kj::HashMap<uint64_t, py::object> unknownEnumerants;
	
	// Type-erased list for faster processing
	capnp::List<uint16_t>::Reader rawData = data.as<AnyList>().as<capnp::List<uint16_t>>();
	
	auto resultHolder = ContiguousCArray::alloc<PyObject*>(shape, "O");
	auto outData = resultHolder.template as<PyObject*>();
	
	auto getEnumerant = [&](uint16_t raw) -> py::object {
		if(raw < knownEnumerants.size())
			return knownEnumerants[raw];
		
		KJ_IF_MAYBE(pEntry, unknownEnumerants.find(raw)) {
			return *pEntry;
		}
		
		py::object newVal = py::cast(EnumInterface(enumSchema, raw));
		unknownEnumerants.insert(raw, newVal);
		return newVal;
	};
	
	for(auto i : kj::indices(rawData)) {
		outData[i] = getEnumerant(rawData[i]).release().ptr();
	}
	
	py::buffer asPyBuffer = py::cast(mv(resultHolder));
	return asPyBuffer.request(true);
}

template<typename T>
py::buffer_info getDataTensor(T tensor) {
	constexpr bool readOnly = !isBuilder<T>();
	
	try {
		// Extract shape and dat
		auto shape = tensor.get("shape").asList().template as<capnp::List<uint64_t>>();
		auto data  = tensor.get("data").asList();
				
		// Extract raw data
		kj::ArrayPtr<const byte> rawBytes = AnyList::Reader(toReader(data.wrapped())).getRawBytes();
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
			(void*) bytesPtr,
			elementSize,
			std::string(formatString.cStr()),
			shape.size(), outShape, strides,
			readOnly
		);
	} catch(kj::Exception& error) {
		KJ_LOG(ERROR, "Failed to create python buffer. See below error\n", error);
	} 
	
	return py::buffer_info((byte*) nullptr, 0);
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
		assign(data, i, py::reinterpret_borrow<py::object>(bufData[i]));
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
		#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
			actualFormat = str(">", actualFormat);
		#else
			actualFormat = str("<", actualFormat);
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

}

bool needsBackReference(DynamicValue::Type t) {
	switch(t) {
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
	
	if(type.which() == capnp::schema::Type::FLOAT64) {
		PyArray_Descr* baseType = PyArray_DescrFromType(NPY_FLOAT64);
		PyArray_Descr* littleEndianType = PyArray_DescrNewByteorder(baseType, NPY_LITTLE);
		Py_DECREF(baseType);
		return littleEndianType;
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

template<typename T>
py::buffer_info scalarFallback(DynamicStructInterface<T> tensor) {
	kj::ArrayPtr<uint64_t> shape = nullptr; // Size 0 array
	
	auto resultHolder = ContiguousCArray::alloc<PyObject*>(shape, "O");
	auto outData = resultHolder.template as<PyObject*>();
	
	DynamicValueType<T> readerOrBuilder(shareMessage(tensor), tensor);
	py::object asObject = py::cast(readerOrBuilder);
	outData[0] = asObject.inc_ref().ptr();
	
	py::buffer asPyBuffer = py::cast(mv(resultHolder));
	return asPyBuffer.request(true);
}

template<typename T>
py::buffer_info getTensorImpl(T tensor) {
	try {
		auto schema = tensor.getSchema();
		auto scalarType = schema.getFieldByName("data").getType().asList().getElementType();
		
		if(scalarType.which() == capnp::schema::Type::ENUM)
			return getEnumTensor(tensor);
		
		if(isObjectType(scalarType)) {
			return getObjectTensor(tensor);
		}
		
		if(scalarType.isBool()) {
			return getBoolTensor(toReader(tensor.wrapped()));
		}
		
		return getDataTensor(tensor);
	} catch(kj::Exception& e) {
		return scalarFallback(tensor);
	}
}

py::buffer_info getTensor(DynamicStructInterface<capnp::DynamicStruct::Reader> reader) {
	return getTensorImpl(reader);
}

py::buffer_info getTensor(DynamicStructInterface<capnp::DynamicStruct::Builder> reader) {
	return getTensorImpl(reader);
}


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

}
