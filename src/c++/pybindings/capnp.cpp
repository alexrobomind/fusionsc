// This translation module is responsible for calling import_array in numpy
#define FSCPY_IMPORT_ARRAY

#include "tensor.h"

#include "fscpy.h"
#include "loader.h"
#include "assign.h"
#include "graphviz.h"
#include "data.h"

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
#include <fsc/yaml.h>

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
	
	if(backReference) {
		fieldValue.attr("_parent") = self;
		// py::print("Installed BR", self, fieldValue.attr("_parent"));
	}
	
	return fieldValue;
}

/**
 * Determines which readers / builders need to store a reference to
 * their parent to keep their data alive.
 */
template<typename T>
bool installBackReference(T&& t, capnp::Type type) {
	// Unconstrained capability types need no back reference because
	// of conversion logic.
	if(t.getType() == capnp::DynamicValue::ANY_POINTER) {
		if(type.isAnyPointer() && type.whichAnyPointerKind() == capnp::schema::Type::AnyPointer::Unconstrained::CAPABILITY)
			return false;
	}
	
	return needsBackReference(t.getType());
}

//! Pipelines have their own keep-alive through PipelineHooks
template<>
bool installBackReference<DynamicValuePipeline>(DynamicValuePipeline&& t, capnp::Type type) {
	return false;
}

//! Pipelines have their own keep-alive through PipelineHooks
template<>
bool installBackReference<DynamicValuePipeline&>(DynamicValuePipeline& t, capnp::Type type) {
	return false;
}

DynamicValuePipeline anyPtrToInterface(DynamicValuePipeline&& pipeline, capnp::InterfaceSchema schema) {
	// The AnyPtr -> interface conversion is done by DynamicStructPipeline::get(...)
	return pipeline;
}

DynamicValue::Reader anyPtrToInterface(DynamicValue::Reader&& reader, capnp::InterfaceSchema schema) {
	capnp::AnyPointer::Reader anyReader = reader.as<capnp::AnyPointer>();
	return anyReader.getAs<capnp::DynamicCapability>(schema.asInterface());
}

DynamicValue::Builder anyPtrToInterface(DynamicValue::Builder&& builder, capnp::InterfaceSchema schema) {
	capnp::AnyPointer::Builder anyBuilder = builder.as<capnp::AnyPointer>();
	return anyBuilder.getAs<capnp::DynamicCapability>(schema.asInterface());
}

/**
 * Implementation for _get used by getField, which determines, based on type
 * dispatch, whether the returned value needs a back-reference.
 */
template<typename T>
py::tuple underscoreGet(T& ds, kj::StringPtr field) {
	auto cppValue = ds.get(field);
	auto fieldType = ds.getSchema().getFieldByName(field).getType();
	bool nbr = installBackReference(cppValue, fieldType);
	
	// Fields of constrained AnyPointer type need to be adjusted
	if(fieldType.isAnyPointer() && fieldType.whichAnyPointerKind() == capnp::schema::Type::AnyPointer::Unconstrained::CAPABILITY) {
		auto schema = defaultLoader.importBuiltin<capnp::Capability>();
		cppValue = anyPtrToInterface(mv(cppValue), schema.asInterface());
	}
	
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
	
	py::class_<DR>(m, "DataReader", py::dynamic_attr(), py::buffer_protocol())
		.def_buffer([](DR& dr) {
			return py::buffer_info(dr.begin(), dr.size(), true);
		})
	;
	py::class_<DB>(m, "DataBuilder", py::dynamic_attr(), py::buffer_protocol())
		.def_buffer([](DB& dr) {
			return py::buffer_info(dr.begin(), dr.size(), false);
		})
	;
	
	py::class_<DP>(m, "DataPipeline", py::dynamic_attr());
}

template<typename List>
struct Iterator {
	py::object pyList;
	List& list;
	size_t pos;
	
	Iterator(py::object pyList, List& list, size_t pos) : pyList(mv(pyList)), list(list), pos(pos) {}
	
	bool operator==(Iterator other) { return pos == other.pos; }
	bool operator!=(Iterator other) { return pos != other.pos; }
	Iterator& operator++() { ++pos; return *this; }
	
	py::object operator*() {
		auto result = list[pos];
		
		py::object pyResult = py::cast(result);
		
		if(installBackReference(result, list.getSchema().getElementType()))
			pyResult.attr("_parent") = pyList;
		
		return pyResult;
	}
};

template<typename T>
void defGetItemAndLen(py::class_<T>& c) {
	//c.def("__getitem__", [](T& list, size_t idx) { return list[idx]; }, py::keep_alive<0, 1>());
	c.def("__getitem__", [](py::object pyList, size_t idx) {
		T& list = py::cast<T&>(pyList);
		
		auto result = list[idx];
		py::object pyResult = py::cast(result);
		
		if(installBackReference(result, list.getSchema().getElementType()))
			pyResult.attr("_parent") = pyList;
		
		return pyResult;		
	});
	c.def("__len__", [](T& list) { return list.size(); });
	c.def("__iter__",
		[](py::object pyList) {
			T& list = py::cast<T&>(pyList);
			
			//TODO: This relies on internals, but the default way looks bugged
			Iterator begin(pyList, list, 0);
			Iterator end(mv(pyList), list, list.size());
			
			return py::make_iterator(begin, end);
		},
		py::keep_alive<0, 1>()
	);
}

void setItem(capnp::DynamicList::Builder list, unsigned int index, py::object value) {
	assign(list, index, mv(value));
}

void setFieldByName(capnp::DynamicStruct::Builder builder, kj::StringPtr fieldName, py::object value) {
	assign(builder, fieldName, mv(value));
}

void setField(capnp::DynamicStruct::Builder builder, capnp::StructSchema::Field field, py::object value) {
	// Note: This is neccessary because for branded structs, the field descriptors are shared by all brands
	// assign(FieldSlot(builder, field), mv(value));
	assign(mv(builder), field.getProto().getName(), mv(value));
}

DynamicValue::Builder initField(capnp::DynamicStruct::Builder builder, capnp::StructSchema::Field field) {
	// Note: This is neccessary because for branded structs, the field descriptors are shared by all brands
	// return builder.init(field);
	return builder.init(field.getProto().getName());
}

DynamicValue::Builder initFieldByName(capnp::DynamicStruct::Builder builder, kj::StringPtr fieldName) {
	return builder.init(fieldName);
}

DynamicValue::Builder initList(capnp::DynamicStruct::Builder builder, capnp::StructSchema::Field field, size_t size) {
	// return builder.init(field, size);
	return builder.init(field.getProto().getName(), size);
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

py::object clone(DynamicStruct::Reader self) {
	auto msg = new capnp::MallocMessageBuilder();
	msg->setRoot(self);
	
	auto pyBuilder = py::cast(capnp::DynamicValue::Builder(msg->getRoot<capnp::DynamicStruct>(self.getSchema())));
	
	auto pyMsg = py::cast(msg);
	pyBuilder.attr("_msg") = pyMsg;
	
	return pyBuilder;
}

kj::String toYaml(DynamicStruct::Reader self, bool flow) {
	auto emitter = kj::heap<YAML::Emitter>();
	
	if(flow) {
		emitter -> SetMapFormat(YAML::Flow);
		emitter -> SetSeqFormat(YAML::Flow);
	}
	
	(*emitter) << self;
	
	kj::ArrayPtr<char> stringData(const_cast<char*>(emitter -> c_str()), emitter -> size() + 1);
	
	return kj::String(stringData.attach(mv(emitter)));
}

kj::String repr(DynamicStruct::Reader self) {
	return toYaml(self, true);
}

kj::String typeStr1(DynamicStruct::Reader reader) {
	capnp::Type type = reader.getSchema();
	Temporary<capnp::schema::Type> typeProto;
	extractType(type, typeProto);
	return kj::str(typeProto.asReader());
}

kj::String typeStr2(DynamicCapability::Client clt) {
	capnp::Type type = clt.getSchema();
	Temporary<capnp::schema::Type> typeProto;
	extractType(type, typeProto);
	return kj::str(typeProto.asReader());
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
		return getTensor(tensor, readOnly);
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
	c.def("_get", [](T& self, capnp::StructSchema::Field field) { return underscoreGet<T>(self, field.getProto().getName()); });
	c.def("_get", &underscoreGet<T>);
	
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

// Pickling support

kj::Array<const byte> fromPythonBuffer(py::buffer buf) {
	auto bufInfo = buf.request();
	
	kj::ArrayPtr<const byte> ptr((const byte*) bufInfo.ptr, bufInfo.itemsize * bufInfo.size);
	
	// Add a deleter for the buffer info that deletes inside the GIL
	Maybe<py::buffer_info> deletable = mv(bufInfo);
	return ptr.attach(kj::defer([deletable = mv(deletable)]() mutable {
		py::gil_scoped_acquire withGil;
		deletable = nullptr;
	}));
}

py::list flattenDataRef(uint32_t pickleVersion, capnp::DynamicCapability::Client dynamicRef) {
	auto payloadType = getRefPayload(dynamicRef.getSchema());
	
	auto data = PythonWaitScope::wait(getActiveThread().dataService().downloadFlat(dynamicRef.castAs<DataRef<>>()));
	
	py::list result(data.size());
	
	if(pickleVersion <= 4) {
		// Version with copying
		for(auto i : kj::indices(data)) {
			result[i] = py::bytes((const char*) data[i].begin(), (uint64_t) data[i].size());
		}
	} else {
		// Zero-copy version
		auto pbCls = py::module_::import("pickle").attr("PickleBuffer");
		for(auto i : kj::indices(data)) {
			py::object asPy = py::cast(capnp::Data::Reader(data[i].asPtr()));
			asPy.attr("_backingArray") = unknownObject(mv(data[i]));
			
			result[i] = pbCls(mv(asPy));
		}
	}
	
	return result;
}

LocalDataRef<> unflattenDataRef(py::list input) {
	auto arrayBuilder = kj::heapArrayBuilder<kj::Array<const byte>>(input.size());
	
	for(auto i : kj::indices(input)) {
		arrayBuilder.add(fromPythonBuffer(py::reinterpret_borrow<py::buffer>(input[i])));
	}
	
	return getActiveThread().dataService().publishFlat<capnp::AnyPointer>(arrayBuilder.finish());
}

void bindPickleRef(py::module_& m, py::class_<capnp::DynamicCapability::Client> cls) {
	// Note: pybind11 generates incorrect qualnames for functions
	// Therefore, we need to define unpicklers and then call them via a
	// wrapper defined by python itself.
	
	m.def("_unpickleRefInner", [](uint32_t pickleVersion, uint32_t version, py::list data) -> capnp::DynamicCapability::Client {
		KJ_REQUIRE(version == 1, "Only version 1 representation supported");
		return unflattenDataRef(data);
	});
	auto unpickler = py::getattr(m, "_unpickleRef");
	
	cls.def("__reduce_ex__", [cls, unpickler](capnp::DynamicCapability::Client src, uint32_t pickleVersion) {
		return py::make_tuple(
			unpickler,
			py::make_tuple(
				pickleVersion,
				1,
				flattenDataRef(pickleVersion, src)
			)
		);
	});
}

void bindPickleReader(py::module_& m, py::class_<capnp::DynamicStruct::Reader> cls) {
	// Note: pybind11 generates incorrect qualnames for functions
	// Therefore, we need to define unpicklers and then call them via a
	// wrapper defined by python itself.
	
	m.def("_unpickleReaderInner", [](uint32_t pickleVersion, uint32_t version, py::list data) {
		KJ_REQUIRE(version == 1, "Only version 1 representation supported");
		auto ref = unflattenDataRef(data);
		return openRef(capnp::schema::Type::AnyPointer::Unconstrained::STRUCT, mv(ref));
	});
	auto unpickler = py::getattr(m, "_unpickleReader");
	
	cls.def("__reduce_ex__", [unpickler, cls](capnp::DynamicStruct::Reader src, uint32_t pickleVersion) mutable {
		return py::make_tuple(
			unpickler,
			py::make_tuple(
				pickleVersion,
				1,
				flattenDataRef(pickleVersion, publishReader(src))
			)
		);
	});
}

void bindPickleBuilder(py::module_& m, py::class_<capnp::DynamicStruct::Builder> cls) {
	// Note: pybind11 generates incorrect qualnames for functions
	// Therefore, we need to define unpicklers and then call them via a
	// wrapper defined by python itself.
	
	m.def("_unpickleBuilderInner", [](uint32_t pickleVersion, uint32_t version, py::list data) {
		KJ_REQUIRE(version == 1, "Only version 1 representation supported");
		auto ref = unflattenDataRef(data);
		
		KJ_IF_MAYBE(pPayloadType, getPayloadType(ref)) {
			auto msgBuilder = kj::heap<capnp::MallocMessageBuilder>();
			msgBuilder -> setRoot(ref.get());
			
			capnp::DynamicStruct::Builder dynamic = msgBuilder -> getRoot<capnp::DynamicStruct>(pPayloadType -> asStruct());
			py::object result = py::cast(dynamic);
			
			result.attr("_msg") = unknownObject(mv(msgBuilder));
			
			return result;
		} else {
			KJ_FAIL_REQUIRE("Payload type missing, can't unpickle");
		}
	});
	
	auto unpickler = py::getattr(m, "_unpickleBuilder");
	cls.def("__reduce_ex__", [unpickler, cls](capnp::DynamicStruct::Builder src, uint32_t pickleVersion) mutable {
		return py::make_tuple(
			unpickler,
			py::make_tuple(
				pickleVersion,
				1,
				flattenDataRef(pickleVersion, publishBuilder(src))
			)
		);
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
	
	cDSB.def("set", &setField);
	cDSB.def("set", &setFieldByName);
	
	//cDSB.def("init", &initField);
	//cDSB.def("init", &initFieldByName);
	//cDSB.def("init", &initList);
	//cDSB.def("init", &initListByName);
		
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
	
	cDSB.def("items", [](py::object pyBuilder) {
		DSB& builder = py::cast<DSB&>(pyBuilder);
		py::list result;
		
		auto schema = builder.getSchema();
		
		for(auto field : schema.getNonUnionFields())
			result.append(py::make_tuple(field.getProto().getName(), getField(pyBuilder, py::cast(field))));
		
		KJ_IF_MAYBE(pField, builder.which()) {
			result.append(py::make_tuple(pField->getProto().getName(), getField(pyBuilder, py::cast(*pField))));
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
	
	cDSB.def("__repr__", &repr);
	
	bindPickleBuilder(m, cDSB);
	
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
	
	cDSR.def("items", [](py::object pyReader) {
		DSR& reader = py::cast<DSR&>(pyReader);
		py::list result;
		
		auto schema = reader.getSchema();
		
		for(auto field : schema.getNonUnionFields())
			result.append(py::make_tuple(field.getProto().getName(), getField(pyReader, py::cast(field))));
		
		KJ_IF_MAYBE(pField, reader.which()) {
			result.append(py::make_tuple(pField->getProto().getName(), getField(pyReader, py::cast(*pField))));
		}
		
		return result;
	});
	
	cDSR.def("__repr__", &repr);
	
	bindPickleReader(m, cDSR);
	
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
	
	cDSP.def("items", [](py::object pyPipeline) {
		DSP& reader = py::cast<DSP&>(pyPipeline);
		
		py::list result;
		
		auto schema = reader.getSchema();
		
		for(auto field : schema.getNonUnionFields())
			result.append(py::make_tuple(field.getProto().getName(), getField(pyPipeline, py::cast(field))));
		
		return result;
	});
	
	
	// ----------------- OTHERS ------------------
		
	py::class_<capnp::Orphan<DynamicValue>>(m, "DynamicOrphan", py::dynamic_attr())
		.def_property_readonly("val", [](capnp::Orphan<DynamicValue>& self) { return self.get(); }, py::keep_alive<0, 1>())
	;
	
	py::class_<DST, DSB> cDST(m, "DynamicMessage", py::metaclass(*baseMetaType));
	
	// Conversion
	
	cDSR.def(py::init([](DSB& dsb) { return dsb.asReader(); }));
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
	py::class_<TrackedMessageBuilder, capnp::MallocMessageBuilder>(m, "TrackedMessageBuilder")
		.def(py::init())
	;
}

void bindCapClasses(py::module_& m) {
	auto cDCC = py::class_<capnp::DynamicCapability::Client>(m, "DynamicCapabilityClient", py::multiple_inheritance(), py::metaclass(*baseMetaType))
		.def(py::init([](capnp::DynamicCapability::Client src) { return src; }))
	;
	
	bindPickleRef(m, cDCC);
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

void bindAnyClasses(py::module_& m) {
	py::class_<capnp::AnyPointer::Builder>(m, "AnyBuilder");
	py::class_<capnp::AnyPointer::Reader>(m, "AnyReader");
	py::class_<capnp::Response<AnyPointer>>(m, "AnyResponse");
}

uint64_t totalSize(capnp::DynamicStruct::Reader reader) {
	return reader.totalSize().wordCount * 8;
}

void bindHelpers(py::module_& m) {
	m.def("totalSize", &totalSize);
	m.def("toYaml", &toYaml, py::arg("readerOrBuilder"), py::arg("flow") = false);
	m.def("clone", &clone, py::arg("cloneFrom"));
	
	m.def("typeStr", &typeStr1);
	m.def("typeStr", &typeStr2);
	
	m.def("prettyPrint", [](capnp::DynamicStruct::Reader& self) {
		return capnp::prettyPrint(self).flatten();
	});
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
	
	defaultLoader.addBuiltin<capnp::Capability>();
	
	py::module_ mcapnp = m.def_submodule("capnp", "Python bindings for Cap'n'proto classes (excluding KJ library)");
	
	py::object scope = mcapnp.attr("__dict__");
	py::exec(
		"def _unpickleReader(pickleVersion, version, data):\n"
		"	return _unpickleReaderInner(pickleVersion, version, data)\n"
		"\n"
		"def _unpickleBuilder(pickleVersion, version, data):\n"
		"	return _unpickleBuilderInner(pickleVersion, version, data)\n"
		"\n"
		"def _unpickleRef(pickleVersion, version, data):\n"
		"	return _unpickleRefInner(pickleVersion, version, data)\n"
		"\n",
		scope
	);
	
	bindListClasses(mcapnp);
	bindBlobClasses(mcapnp);
	bindStructClasses(mcapnp);
	bindFieldDescriptors(mcapnp);
	bindMessageBuilders(mcapnp);
	bindCapClasses(mcapnp);
	bindEnumClasses(mcapnp);
	bindSchemaClasses(mcapnp);
	bindHelpers(mcapnp);
	bindAnyClasses(mcapnp);
	
	m.add_object("void", py::cast(capnp::DynamicValue::Reader(capnp::Void())));
	
	m.def("visualize", &visualizeGraph);
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

DynamicValuePipeline DynamicStructPipeline::get(kj::StringPtr name) {
	KJ_IF_MAYBE(pField, schema.findFieldByName(name)) {
		return get(*pField);
	} else {
		KJ_FAIL_REQUIRE("Field not found", name);
	}
}

}