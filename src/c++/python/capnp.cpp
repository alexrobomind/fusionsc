#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>

#include <kj/async.h>
#include <kj/mutex.h>

#include <capnp/dynamic.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/blob.h>

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

py::object getField(py::object self, py::object field) {
	py::tuple getResult = self.attr("_get")(field);
	
	py::object fieldValue  = getResult[py::cast(0)];
	bool backReference = py::cast<bool>(getResult[1]);
	
	if(backReference)
		fieldValue.attr("_parent") = self;
	
	return fieldValue;
}

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

// Pipelines count references themselves
template<>
bool needsBackReference<DynamicValuePipeline>(DynamicValuePipeline& t) {
	return false;
}

template<typename T, typename Field>
py::tuple underscoreGet(T& ds, Field field) {
	auto cppValue = ds.get(field);
	bool nbr = needsBackReference(cppValue);
	
	return py::make_tuple(
		py::cast(mv(cppValue)),
		py::cast(nbr)
	);
};


void bindBlobClasses(py::module_& m) {
	using TR = capnp::Text::Reader;
	using TB = capnp::Text::Builder;
	using TP = capnp::Text::Pipeline;
	
	using AP = kj::ArrayPtr<kj::byte>;
	using CAP = kj::ArrayPtr<const kj::byte>;
	
	py::class_<TR, kj::StringPtr>(m, "TextReader", py::dynamic_attr());
	py::class_<TB>(m, "TextBuilder", py::dynamic_attr())
		.def("__repr__", [](TB& self) { return self.asString(); })
	;
	py::class_<TP>(m, "TextPipeline", py::dynamic_attr());
	
	using DR = capnp::Data::Reader;
	using DB = capnp::Data::Builder;
	using DP = capnp::Data::Pipeline;
	
	py::class_<DR, CAP>(m, "DataReader", py::dynamic_attr());
	py::class_<DB, AP>(m, "DataBuilder", py::dynamic_attr());
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

template<typename T>
void defBuffer(py::class_<T>& c, bool readOnly) {
	c.def_buffer([readOnly](T& list) {
		capnp::AnyList::Reader anyReader = list;
		
		kj::ArrayPtr<const byte> rawBytes = list.getRawBytes();
		byte* bytesPtr = const_cast<byte*> rawList.begin();
		
		size_t elementSize = 0;
		const char* formatString = "";
		
		switch(list.getType().which()) {
			case capnp::schema::Type::VOID:
			case capnp::schema::Type::BOOL:
			case capnp::schema::Type::TEXT:
			case capnp::schema::Type::DATA:
			case capnp::schema::Type::LIST:
			case capnp::schema::Type::STRUCT:
			case capnp::schema::Type::INTERFACE:
			case capnp::schema::Type::ANY_POINTER:
				KJ_FAIL_REQUIRE("Can not access list of non-promitive data type.", list.getType().which());
			
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
		}
		
		switch(list.getElementSize()) {
			case capnp::ElementSize::BYTE:
			case capnp::ElementSize::TWO_BYTES:
			case capnp::ElementSize::FOUR_BYTES:
			case capnp::ElementSize::EIGHT_BYTES:
				KJ_REQUIRE(elementSize * list.size() == rawBytes.size());
			
			case capnp::ElementSize::VOID:
			case capnp::ElementSize::BIT:
				KJ_FAIL_REQUIRE("VOID and BIT lists may not be accessed directly");
			
			case capnp::ElementSize::POINTER:
				KJ_FAIL_REQUIRE("Pointer lists may not be accessed directly");
				
			case capnp::ElementSize::INLINE_COMPOSITE:
				KJ_FAIL_REQUIRE("Struct lists may not (yet) be accessed directly");
		}
		
		return py::buffer_info(
			(void*) bytesPtr, elementSize, std::string(formatString), list.size(), readOnly
		);
	});
}


template<typename T>
void defSetItem(py::class_<T>& c) {
	c.def("__setitem__", [](T& list, size_t idx, DynamicValue::Reader value) {	
		list.set(idx, value);
	});
}

void bindListClasses(py::module_& m) {
	using DLB = DynamicList::Builder;
	using DLR = DynamicList::Reader;
	
	py::class_<DLB> cDLB(m, "DynamicListBuilder", py::dynamic_attr());
	defGetItemAndLen(cDLB);
	defSetItem(cDLB);
	
	py::class_<DLR> cDLR(m, "DynamicListReader", py::dynamic_attr());
	defGetItemAndLen(cDLR);
}

template<typename T, typename... Extra>
void defGet(py::class_<T, Extra...>& c) {
	c.def("_get", [](T& self, capnp::StructSchema::Field field) { return underscoreGet<T, kj::StringPtr>(self, field.getProto().getName()); });
	c.def("_get", &underscoreGet<T, kj::StringPtr>);
	
	auto genericGet = [](py::object ds, py::object field) { return getField(ds, field); };
	
	c.def("get", genericGet);
	c.def("__getitem__", genericGet);
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
	
	py::class_<DSB> cDSB(m, "DynamicStructBuilder", py::dynamic_attr(), py::multiple_inheritance(), py::metaclass(*baseMetaType));
	cDSB.def(py::init([](DynamicStruct::Builder b) { return b; }));
	
	defGet(cDSB);
	defHas(cDSB);
	defWhich(cDSB);
	
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	cDSB.def("__setitem__", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("__setitem__", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	cDSB.def("__delitem__", [](DSB& dsb, kj::StringPtr name) { dsb.clear(name); });
	
	cDSB.def("__len__", [](DSB& ds) {
		auto schema = ds.getSchema();
		size_t result = schema.getNonUnionFields().size();
		if(schema.getUnionFields().size() > 0)
			result += 1;
		
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
	
	py::class_<DST, DSB> cDST(m, "DynamicMessage", py::metaclass(*baseMetaType));
	
	py::class_<DSR> cDSR(m, "DynamicStructReader", py::dynamic_attr(), py::multiple_inheritance(), py::metaclass(*baseMetaType));
	cDSR.def(py::init([](DynamicStruct::Reader r) { return r; }));
	
	defGet(cDSR);
	defHas(cDSR);
	defWhich(cDSR);
	
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
	
	py::class_<capnp::Response<AnyPointer>>(m, "DynamicResponse");
}

void bindFieldDescriptors(py::module_& m) {
	py::class_<capnp::StructSchema::Field>(m, "Field")		
		.def("__get__", [](capnp::StructSchema::Field& field, py::object self, py::object type) -> py::object {
			if(self.is_none())
				return py::cast(kj::str("Field ", field.getProto().getName(), " : ", typeName(field.getType())));
			
			return getField(self, py::cast(field));
		})
		
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, DynamicValue::Reader value) { self.set(field, value); })
		.def("__delete__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self) { self.clear(field); })
	;
}

void bindMessageBuilders(py::module_& m) {
	py::class_<capnp::MessageBuilder>(m, "MessageBuilder");
	py::class_<capnp::MallocMessageBuilder>(m, "MallocMessageBuilder");
}

void bindCapClasses(py::module_& m) {
	py::class_<capnp::DynamicCapability::Client>(m, "DynamicCapabilityClient", py::multiple_inheritance(), py::metaclass(*baseMetaType))
		.def(py::init([](capnp::DynamicCapability::Client src) { return src; }))
	;
	py::class_<capnp::DynamicCapability::Server>(m, "DynamicCapabilityServer", py::multiple_inheritance(), py::metaclass(*baseMetaType));
}

}

namespace fscpy {
	
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

void bindCapnpClasses(py::module_& m) {
	py::module_ mcapnp = m.def_submodule("capnp");
	
	bindListClasses(mcapnp);
	bindBlobClasses(mcapnp);
	bindStructClasses(mcapnp);
	bindFieldDescriptors(mcapnp);
	bindMessageBuilders(mcapnp);
	bindCapClasses(mcapnp);
	
	m.add_object("void", py::cast(capnp::DynamicValue::Reader(capnp::Void())));
}

}