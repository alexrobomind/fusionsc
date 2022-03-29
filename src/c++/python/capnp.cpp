#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>

#include <kj/async.h>
#include <kj/mutex.h>

#include <capnp/dynamic.h>
#include <capnp/blob.h>

#include <fsc/local.h>
#include <fsc/common.h>

#include "dynamic_value.h"

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

namespace py = pybind11;

namespace fsc {

struct PyContext {
	static Library library() {
		{
			auto locked = _library.lockShared();
			
			if(locked->get() != nullptr)
				return locked->get()->addRef();
		}
		
		auto locked = _library.lockExclusive();
		*locked = newLibrary();
		
		return locked->get()->addRef();
	}
	
	static LibraryThread libraryThread() {
		ensureLT();
		return _libraryThread->addRef();
	}
	
	static kj::WaitScope& waitScope() {
		ensureLT();
		return _libraryThread -> waitScope();
	}
	
private:
	static kj::MutexGuarded<Library> _library;
	static thread_local LibraryThread _libraryThread;
	
	static void ensureLT() {
		if(_libraryThread.get() == nullptr) {
			_libraryThread = library() -> newThread();
		}
	}
};

struct PyPromise {
	PyPromise(Promise<py::object> input) :
		holder(input.fork())
	{}
	
	PyPromise(PyPromise& other) :
		holder(other.holder.addBranch().fork())
	{}
	
	Promise<py::object> get() {
		return holder.addBranch();
	}
	
	operator Promise<py::object>() {
		return get();
	}
	
	py::object wait() {
		py::gil_scoped_release release_gil;
		return holder.addBranch().wait(PyContext::waitScope());
	}
	
	bool poll() {
		py::gil_scoped_release release_gil;
		return holder.addBranch().poll(PyContext::waitScope());
	}
	
private:
	kj::ForkedPromise<py::object> holder;
};

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
	
	py::class_<DR, CAP>(m, "TextReader");
	py::class_<DB, AP>(m, "TextBuilder");
	py::class_<DP>(m, "TextPipeline");
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

void dynamicStructClasses(py::module_ m) {
	using DSB = DynamicStruct::Builder;
	using DSR = DynamicStruct::Reader;
	using DSP = DynamicStruct::Pipeline;
	
	py::class_<DSB> cDSB(m, "DynamicStructBuilder");
	defGet(cDSB);
	defHas(cDSB);
	
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); });
	cDSB.def("set", [](DSB& dsb, kj::StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); });
	
	py::class_<DSR> cDSR(m, "DynamicStructBuilder");
	defGet(cDSR);
	defHas(cDSR);
	
	py::class_<DSP> cDSP(m, "DynamicStructPipeline");
	defGet(cDSP);
}

}