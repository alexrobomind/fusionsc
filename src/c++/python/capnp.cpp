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
#include "dynamic_value.h"
#include "async.h"

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

namespace py = pybind11;

// Definitions local to translation unit

namespace fscpy { namespace {

void blobClasses(py::module_ m) {
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

void listClasses(py::module_ m) {
	using DLB = DynamicList::Builder;
	using DLR = DynamicList::Reader;
	
	py::class_<DLB> cDLB(m, "DynamicListBuilder");
	defGetItem(cDLB);
	defSetItem(cDLB);
	
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

void structClasses(py::module_ m) {
	using DSB = DynamicStruct::Builder;
	using DSR = DynamicStruct::Reader;
	using DSP = DynamicStruct::Pipeline;
	
	py::class_<DSB> cDSB(m, "DynamicStructBuilder");
	defGet(cDSB);
	defHas(cDSB);
	
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	
	py::class_<DSR> cDSR(m, "DynamicStructReader");
	defGet(cDSR);
	defHas(cDSR);
	
	py::class_<DSP> cDSP(m, "DynamicStructPipeline");
	defGet(cDSP);
}

}}

namespace fscpy {

void defCapnpClasses(py::module_ m) {
	listClasses(m);
	blobClasses(m);
	structClasses(m);
}

}