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

void bindBlobClasses(py::module_& m) {
	using TR = capnp::Text::Reader;
	using TB = capnp::Text::Builder;
	using TP = capnp::Text::Pipeline;
	
	using AP = kj::ArrayPtr<kj::byte>;
	using CAP = kj::ArrayPtr<const kj::byte>;
	
	py::class_<TR, kj::StringPtr>(m, "TextReader");
	py::class_<TB>(m, "TextBuilder");
	py::class_<TP>(m, "TextPipeline");
	
	using DR = capnp::Data::Reader;
	using DB = capnp::Data::Builder;
	using DP = capnp::Data::Pipeline;
	
	py::class_<DR, CAP>(m, "DataReader");
	py::class_<DB, AP>(m, "DataBuilder");
	py::class_<DP>(m, "DataPipeline");
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
	
	py::class_<DLB> cDLB(m, "DynamicListBuilder");
	defGetItem(cDLB);
	defSetItem(cDLB);
	
	py::class_<DLR> cDLR(m, "DynamicListReader");
	defGetItem(cDLR);
}

template<typename T, typename... Extra>
void defGet(py::class_<T, Extra...>& c) {
	c.def("get", [](T& ds, kj::StringPtr name) { return ds.get(name); }, py::keep_alive<0, 1>());
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
	
	py::class_<DSB> cDSB(m, "DynamicStructBuilder", py::dynamic_attr(), py::multiple_inheritance());
	cDSB.def(py::init([](DynamicStruct::Builder b) { return b; }));
	
	defGet(cDSB);
	defHas(cDSB);
	defWhich(cDSB);
	
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	
	py::class_<DST, DSB> cDST(m, "DynamicMessage");
	
	py::class_<DSR> cDSR(m, "DynamicStructReader", py::dynamic_attr(), py::multiple_inheritance());
	cDSR.def(py::init([](DynamicStruct::Reader r) { return r; }));
	
	defGet(cDSR);
	defHas(cDSR);
	defWhich(cDSR);
	
	py::class_<DSP> cDSP(m, "DynamicStructPipeline", py::multiple_inheritance());
	defGet(cDSP);
	
	// TODO: This is a little bit hacky, but currently the only way we can allow the construction of derived instances
	cDSP.def(py::init([](DynamicStruct::Pipeline& p, kj::StringPtr key) {
		KJ_REQUIRE((key == INTERNAL_ACCESS_KEY), "The pipeline constructor is reserved for internal use");
		py::print("Moving pipeline"); 
		return kj::mv(p);
	}));
	
	py::class_<capnp::Response<DynamicStruct>, DSR>(m, "DynamicResponse");
}

void bindFieldDescriptors(py::module_& m) {
	py::class_<capnp::StructSchema::Field>(m, "Field")
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Pipeline& self, py::object cls) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Reader& self, py::object cls) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, py::object cls) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		
		.def("__get__", [](capnp::StructSchema::Field& field, py::object self, py::object type) { return kj::str("Field ", field.getProto().getName(), " : ", typeName(field.getType())); })
		
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, DynamicValue::Reader value) { self.set(field, value); })
		.def("__delete__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self) { self.clear(field); })
	;
}

void bindMessageBuilders(py::module_& m) {
	py::class_<capnp::MessageBuilder>(m, "MessageBuilder");
	py::class_<capnp::MallocMessageBuilder>(m, "MallocMessageBuilder");
}

void bindCapClasses(py::module_& m) {
	py::class_<capnp::DynamicCapability::Client>(m, "DynamicCapabilityClient", py::multiple_inheritance())
		.def(py::init([](capnp::DynamicCapability::Client src) { return src; }))
	;
	py::class_<capnp::DynamicCapability::Server>(m, "DynamicCapabilityServer", py::multiple_inheritance());
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