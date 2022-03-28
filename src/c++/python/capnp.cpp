#include <pybind11/pybind11.h>

#include <kj/async.h>

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;

namespace fsc {

struct PyContext {
	static Library library() {
		{
			auto locked = _library.lockShared();
			
			if(locked->get() != nullptr)
				return locked->addRef();
		}
		
		auto locked = _library.lockExclusive();
		*locked = newLibrary();
		
		return locked->addRef();
	}
	
	static LibraryThread libraryThread() {
		ensureLT();
		return _libraryThread->addRef();
	}
	
	static WaitScope& waitScope() {
		ensureLT();
		return _libraryThread -> waitScope();
	}
	
private:
	static MutexGuarded<Library> _library;
	static thread_local LibraryThread _libraryThread;
	
	static void ensureLT() {
		if(_libraryThread.get() == nullptr) {
			_libraryThread = library() -> newThread();
		}
	}
};

struct PyPromise {
	PyPromise(Promise<py::object> input) :
		storage(input.fork())
	{}
	
	Promise<py::object> get() {
		return holder.addBranch();
	}
	
	operator Promise<py::object>() {
		return get();
	}
	
	py::object wait() {
		return holder.addBranch().wait(PyContext::waitScope());
	}
	
	bool poll() {
		return holder.addBranch().poll(PyContext::waitScope());
	}
	
private:
	kj::ForkedPromise<py::object> holder;
};

// Shifts the type erasure into the pybind11 layer
template<typename T>
py::object dynamicToPy(T input) {
	auto type = input.getType();
	
	#DEFINE FSC_DC(EnumVal, Type) \
		if(type == DynamicValue::EnumVal) \
			return py::cast(input.as<Type>());
			
	FSC_DC(VOID, int);
	FSC_DC(BOOL, bool);
	FSC_DC(INT, int);
	FSC_DC(UINT, unsigned int);
	FSC_DC(FLOAT, float);
	FSC_DC(TEXT, capnp::Text);
	FSC_DC(DATA, capnp::Data);
	FSC_DC(LIST, capnp::DynamicList);
	FSC_DC(ENUM, capnp::DynamicEnum);
	FSC_DC(STRUCT, capnp::DynamicStruct);
	FSC_DC(CAPABILITY, capnp::DynamicCapability);
	FSC_DC(ANY_POINTER, capnp::AnyPointer);
	
	#undef FSC_DC
	
	KJ_FAIL_REQUIRE("Unknown type", type);
}

void blobClasses(py::module_ m) {
	using TR = capnp::Text::Reader;
	using TB = capnp::Text::Builder;
	using TP = capnp::Text::Pipeline;
	
	using AP = kj::ArrayPtr<kj::byte>;
	using CAP = kj::ArrayPtr<const kj::byte>;
	
	py::class_<TR, CAP>("TextReader");
	py::class_<TB, AP>("TextBuilder");
	py::class_<TP>("TextPipeline");
	
	using DR = capnp::Data::Reader;
	using DB = capnp::Data::Builder;
	using DP = capnp::Data::Pipeline;
	
	py::class_<DR, CAP>("TextReader");
	py::class_<DB, AP>("TextBuilder");
	py::class_<DP>("TextPipeline");
}

template<typename T>
void defGetItem(py::class_<T>& c) {
	c.def("__getitem__", [](T& list, size_t idx) { return dynamicToPy(list[idx]); });
}


template<typename T>
void defSetItem(py::class_<T>& c) {
	c.def("__setitem__", [](T& list, size_t idx, py::object value) {
		auto asReader = value.cast<DynamicValue::Reader*>();
		if(asReader != nullptr)
			list.set(idx, *asReader);
		
		auto asBuilder = value.cast<DynamicValue::Builder*>();
		if(asBuilder != nullptr)
			list.set(idx, *asBuilder);
		
	});
}

void listClasses(py::module_ m) {
}

template<typename T>
void defGet(py::class_<T>& c) {
	c.def("get", [](T& ds, StringPtr name) { return dynamicToPy(ds.get(name)); }, py::keep_alive<0, 1>());
}

template<typename T>
void defHas(py::class_<T>& c) {
	c.def("has", [](T& ds, StringPtr name) { return ds.has(name); });
}

void dynamicStructClasses(py::module_ m) {
	using DSB = DynamicStruct::Builder;
	using DSR = DynamicStruct::Reader;
	using DSP = DynamicStruct::Pipeline;
	
	py::class_<DSB>(m, "DynamicStructBuilder") cDSB;
	defGet(cDSB);
	defHas(cDSB);
	
	cDSB.def("set", [](DSB& dsb, StringPtr name, const DynamicValue::Reader& val) { dsb.set(name, val); };
	cDSB.def("set", [](DSB& dsb, StringPtr name, const DynamicValue::Builder& val) { dsb.set(name, val.asReader()); };
	
	py::class_<DSR>(m, "DynamicStructBuilder") cDSR;
	defGet(cDSR);
	defHas(cDSR);
	
	py::class_<DSP>(m, "DynamicStructPipeline") cDSP;
	defGet(cDSP);
}

}