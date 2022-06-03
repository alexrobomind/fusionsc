#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>

#include <kj/async.h>
#include <kj/mutex.h>

#include <capnp/dynamic.h>
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

template<typename T, typename Field>
py::tuple underscoreGet(T& ds, Field& field) {
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
	py::class_<TB>(m, "TextBuilder", py::dynamic_attr());
	py::class_<TP>(m, "TextPipeline", py::dynamic_attr());
	
	using DR = capnp::Data::Reader;
	using DB = capnp::Data::Builder;
	using DP = capnp::Data::Pipeline;
	
	py::class_<DR, CAP>(m, "DataReader", py::dynamic_attr());
	py::class_<DB, AP>(m, "DataBuilder", py::dynamic_attr());
	py::class_<DP>(m, "DataPipeline", py::dynamic_attr());
}

template<typename T>
void defGetItem(py::class_<T>& c) {
	c.def("__getitem__", [](T& list, size_t idx) { return list[idx]; });
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
	defGetItem(cDLB);
	defSetItem(cDLB);
	
	py::class_<DLR> cDLR(m, "DynamicListReader", py::dynamic_attr());
	defGetItem(cDLR);
}

template<typename T, typename... Extra>
void defGet(py::class_<T, Extra...>& c) {
	c.def("_get", &underscoreGet<T, capnp::StructSchema::Field>);
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
	using DSP = DynamicStruct::Pipeline;
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
	cDSP.def(py::init([](DynamicStruct::Pipeline& p, kj::StringPtr key) {
		KJ_REQUIRE((key == INTERNAL_ACCESS_KEY), "The pipeline constructor is reserved for internal use");
		py::print("Moving pipeline"); 
		return kj::mv(p);
	}));
	
	cDSP.def("__len__", [](DSP& ds) {
		auto schema = ds.getSchema();
		size_t result = schema.getNonUnionFields().size();
		
		return result;
	});
	
	cDSP.def("__iter__", [](DSP& ds) {
		py::list result;
		
		for(auto field : ds.getSchema().getNonUnionFields()) {
			result.append(str(field.getProto().getName()));
		}
		
		return py::eval("iter")(result);
	});
	
	py::class_<capnp::Response<DynamicStruct>, DSR>(m, "DynamicResponse");
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

void bindCapnpClasses(py::module_& m) {
	py::module_ mcapnp = m.def_submodule("capnp");
	
	bindListClasses(mcapnp);
	bindBlobClasses(mcapnp);
	bindStructClasses(mcapnp);
	bindFieldDescriptors(mcapnp);
	bindMessageBuilders(mcapnp);
	bindCapClasses(mcapnp);
}

}