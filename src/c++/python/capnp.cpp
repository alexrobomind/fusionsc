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
bool needsBackReference(T& t) {
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
bool needsBackReference<DynamicValuePipeline>(DynamicValuePipeline& t) {
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

Tuple<size_t, kj::StringPtr> pyFormat(capnp::Type type) {
	size_t elementSize = 0;
	kj::StringPtr formatString;
	
	switch(type.which()) {
		case capnp::schema::Type::VOID:
		case capnp::schema::Type::BOOL:
		case capnp::schema::Type::TEXT:
		case capnp::schema::Type::DATA:
		case capnp::schema::Type::LIST:
		case capnp::schema::Type::STRUCT:
		case capnp::schema::Type::INTERFACE:
		case capnp::schema::Type::ANY_POINTER:
			KJ_FAIL_REQUIRE("Can not format pointer types (text, data, list, struct, capability / interface, anypointer), void and bool.", type.which());
		
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
		case capnp::schema::Type::ENUM:
			KJ_LOG(WARNING, "Currently enums are handled as 16bit unsigned integers. This might become more sophisticated in the future");
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
	}
	
	return tuple(elementSize, formatString);
}

capnp::AnyList::Reader asAnyListReader(capnp::DynamicList::Reader reader) {
	return reader;
}

capnp::AnyList::Reader asAnyListReader(capnp::DynamicList::Builder builder) {
	return asAnyListReader(builder.asReader());
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
	c.def_buffer([readOnly](T& tensor) {
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
	});
}

//! Allows assigning tensor types from a buffer
void setTensor(DynamicStruct::Builder dsb, py::buffer buffer) {	
	py::buffer_info bufinfo = buffer.request();
	
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

void setTensor(DynamicValue::Builder dvb, py::buffer buffer) {
	//TODO: Derive tensor type from buffer value?
	setTensor(dvb.as<DynamicStruct>(), buffer);
}


template<typename T>
void defSetItem(py::class_<T>& c) {
	c.def("__setitem__", [](T& list, size_t idx, DynamicValue::Reader value) {	
		list.set(idx, value);
	});
}

//! Binds capnp::DynamicList::{Builder, Reader}
void bindListClasses(py::module_& m) {
	using DLB = DynamicList::Builder;
	using DLR = DynamicList::Reader;
	
	py::class_<DLB> cDLB(m, "DynamicListBuilder", py::dynamic_attr(), py::buffer_protocol());
	defGetItemAndLen(cDLB);
	defSetItem(cDLB);
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
	
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, capnp::Orphan<DynamicValue>& val) { dsb.adopt(name, mv(val)); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, py::sequence seq) {
		auto val = dsb.init(name, seq.size()).as<DynamicList>();
		for(size_t i = 0; i < seq.size(); ++i)
			val.set(i, py::cast<DynamicValue::Reader>(seq[i]));
	});
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, py::buffer buf) { setTensor(dsb.init(name), buf); });
	cDSB.def("setAs", [](DSB& dsb, py::buffer buf) { setTensor(dsb, buf); });
		
	cDSB.def("__setitem__", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("__setitem__", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	cDSB.def("__setitem__", [](DSB& dsb, kj::StringPtr name, capnp::Orphan<DynamicValue>& val) { dsb.adopt(name, mv(val)); });
	cDSB.def("__setitem__", [](DSB& dsb, kj::StringPtr name, py::sequence seq) {
		auto val = dsb.init(name, seq.size()).as<DynamicList>();
		for(size_t i = 0; i < seq.size(); ++i)
			val.set(i, py::cast<DynamicValue::Reader>(seq[i]));
	});
	cDSB.def("__setitem__", [](DSB& dsb, kj::StringPtr name, py::buffer buf) { setTensor(dsb.init(name), buf); });
		
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
}

void bindFieldDescriptors(py::module_& m) {
	py::class_<capnp::StructSchema::Field>(m, "Field")		
		.def("__get__", [](capnp::StructSchema::Field& field, py::object self, py::object type) -> py::object {
			if(self.is_none())
				return py::cast(kj::str("Field ", field.getProto().getName(), " : ", typeName(field.getType())));
			
			return getField(self, py::cast(field));
		})
		
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, DynamicValue::Reader value) { self.set(field, value); })
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, capnp::Orphan<DynamicValue>& orphan) { self.adopt(field, mv(orphan)); })
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, py::list list) {
			auto val = self.init(field, list.size()).as<DynamicList>();
			for(size_t i = 0; i < list.size(); ++i)
				val.set(i, py::cast<DynamicValue::Reader>(list[i]));
		})
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, py::buffer buf) { setTensor(self.init(field), buf); })
		
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
	;
}

}

namespace fscpy {
	
// init method

void initCapnp(py::module_& m) {
	py::module_ mcapnp = m.def_submodule("capnp");
	
	bindListClasses(mcapnp);
	bindBlobClasses(mcapnp);
	bindStructClasses(mcapnp);
	bindFieldDescriptors(mcapnp);
	bindMessageBuilders(mcapnp);
	bindCapClasses(mcapnp);
	bindEnumClasses(mcapnp);
	
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