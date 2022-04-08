#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>

#include <kj/async.h>
#include <kj/mutex.h>

#include <capnp/dynamic.h>
#include <capnp/blob.h>

#include <fsc/local.h>
#include <fsc/common.h>

#include "fscpy.h"

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

namespace py = pybind11;

// Definitions local to translation unit

namespace fscpy { namespace {

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
	
	KJ_LOG(WARNING, "Binding list builder");
	py::class_<DLB> cDLB(m, "DynamicListBuilder");
	KJ_LOG(WARNING, "Binding setitem");
	defGetItem(cDLB);
	KJ_LOG(WARNING, "Binding getitem");
	defSetItem(cDLB);
	
	KJ_LOG(WARNING, "Binding list reader");
	py::class_<DLR> cDLR(m, "DynamicListReader");
	defGetItem(cDLR);
}

template<typename T>
void defGet(py::class_<T>& c) {
	c.def("get", [](T& ds, kj::StringPtr name) { return ds.get(name); }, py::keep_alive<0, 1>());
}

template<typename T>
void defHas(py::class_<T>& c) {
	c.def("has", [](T& ds, kj::StringPtr name) { return ds.has(name); });
}

template<typename T>
void defWhich(py::class_<T>& c) {
	c.def("which", [](T& ds) {
		auto maybeField = ds.which();
		
		KJ_IF_MAYBE(pField, maybeFiels) {
			return pField->getProto().getName();
		}
		
		return py::none();
	}
}

template<typename T>
void defInit(py::class_<T>& c) {
	c.def(py::init([](T&& other) { return kj::mv(other); }))
}

void bindStructClasses(py::module_& m) {
	using DSB = DynamicStruct::Builder;
	using DSR = DynamicStruct::Reader;
	using DSP = DynamicStruct::Pipeline;
	
	py::class_<DSB> cDSB(m, "DynamicStructBuilder");
	defGet(cDSB);
	defHas(cDSB);
	defWhich(cDSB);
	
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	
	py::class_<DSR> cDSR(m, "DynamicStructReader");
	defGet(cDSR);
	defHas(cDSR);
	defWhich(cDSR);
	
	py::class_<DSP> cDSP(m, "DynamicStructPipeline");
	defGet(cDSP);
}

void bindFieldDescriptors(py::module_& m) {
	py::class_<capnp::StructSchema::Field>(m, "Field")
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Pipeline& self, py::object type) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Reader& self, py::object type) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, py::object type) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, DynamicValue::Reader value) { self.set(field, value); })
		.def("__delete__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self) { self.clear(field); })
	;
}

}}

namespace fscpy {

void bindCapnpClasses(py::module_& m) {
	KJ_LOG(WARNING, "Binding list classes");
	bindListClasses(m);
	KJ_LOG(WARNING, "Binding blob classes");
	bindBlobClasses(m);
	KJ_LOG(WARNING, "Binding struct classes");
	bindStructClasses(m);
	KJ_LOG(WARNING, "Binding field classes");
	bindFieldDescriptors(m);
}

}