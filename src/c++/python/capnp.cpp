#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>

#include <kj/async.h>
#include <kj/mutex.h>

#include <capnp/dynamic.h>
#include <capnp/blob.h>

#include <fsc/local.h>
#include <fsc/common.h>

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

namespace py = pybind11;

// Conversion of capnp::DynamicValue to python objects
static py::dict globalBuilderClasses;
static py::dict globalReaderClasses;

namespace pybind11 { namespace detail {
	
	template<>
	struct type_caster<DynamicValue::Builder> {
		PYBIND11_TYPE_CASTER(DynamicValue::Builder, const_name("DynamicValueBuilder"));
		
		bool load(handle src, bool convert) {			
			object isInstance = py::eval("isinstance");
			if(isInstance(src, py::type::of<DynamicStruct::Builder>())) {
				value = src.cast<DynamicStruct::Builder>();
				return true;
			}
			
			if(isInstance(src, py::type::of<DynamicList::Builder>())) {
				value = src.cast<DynamicList::Builder>();
				return true;
			}
			
			if(isInstance(src, py::type::of<DynamicEnum>())) {
				value = src.cast<DynamicEnum>();
				return true;
			}
			
			if(isInstance(src, py::type::of<DynamicCapability::Client>())) {
				value = src.cast<DynamicCapability::Client>();
				return true;
			}
			
			if(isInstance(src, py::type::of<AnyPointer::Builder>())) {
				value = src.cast<AnyPointer::Builder>();
				return true;
			}
			
			return false;
		}
		
		// TODO: Implement to-python conversion
		/*
			
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
	FSC_DC(ANY_POINTER, capnp::AnyPointer);*/
		
		static handle cast(DynamicValue::Builder src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::VOID: return py::none();
				case DynamicValue::BOOL: return py::cast(src.as<bool>());
				case DynamicValue::INT: return py::cast(src.as<signed long long>());
				case DynamicValue::UINT: return py::cast(src.as<unsigned long long>());
				case DynamicValue::FLOAT: return py::cast(src.as<double>());
				case DynamicValue::TEXT: return py::cast(src.as<capnp::Text>());
				case DynamicValue::DATA: return py::cast(src.as<capnp::Data>());
				case DynamicValue::LIST: return py::cast(src.as<capnp::DynamicList>());
				case DynamicValue::ENUM: return py::cast(src.as<capnp::DynamicEnum>());
				// case DynamicValue::STRUCT: return py::cast(src.as<capnp::DynamicStruct>());
				case DynamicValue::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>());
				case DynamicValue::ANY_POINTER: return py::cast(src.as<capnp::AnyPointer>());
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT);
			
			DynamicStruct::Builder dynamicStruct = src.as<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalBuilderClasses.contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = globalBuilderClasses[py::cast(typeId)];
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				auto result = targetClass(dynamicStruct);
				return result;
			}
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Builder>::cast((DynamicStruct::Builder &&) src, policy, parent);
		}
	};
	
	template<>
	struct type_caster<DynamicValue::Reader> {
		PYBIND11_TYPE_CASTER(DynamicValue::Reader, const_name("DynamicValueReader"));
		
		// If we get a string, we need to store it temporarily
		type_caster<char> strCaster;		
		
		bool load(handle src, bool convert) {
			object pyType = py::eval("type")(src);
			
			if(pyType == py::eval("real")) {
				value = src.cast<double>();
				return true;
			}
			
			if(pyType == py::eval("int")) {
				if(src >= py::eval("0")) {
					value = src.cast<unsigned long long>();
				} else {
					value = src.cast<signed long long>();
				}
				return true;
			}
			
			if(pyType == py::eval("str")) {
				strCaster.load(src, false);
				value = capnp::Text::Reader((char*) strCaster);
				return true;
			}
			
			object isInstance = py::eval("isinstance");
			if(isInstance(src, py::type::of<DynamicStruct::Reader>())) {
				value = src.cast<DynamicStruct::Reader>();
				return true;
			}
			
			if(isInstance(src, py::type::of<DynamicList::Reader>())) {
				value = src.cast<DynamicList::Reader>();
				return true;
			}
			
			if(isInstance(src, py::type::of<DynamicEnum>())) {
				value = src.cast<DynamicEnum>();
				return true;
			}
			
			if(isInstance(src, py::type::of<AnyPointer::Reader>())) {
				value = src.cast<AnyPointer::Reader>();
				return true;
			}
			
			if(isInstance(src, py::type::of<DynamicCapability::Client>())) {
				value = src.cast<DynamicCapability::Client>();
				return true;
			}
			
			type_caster<DynamicValue::Builder> builderCaster;
			if(builderCaster.load(src, convert)) {
				value = ((DynamicValue::Builder& ) builderCaster).asReader();
				return true;
			}
			
			return false;
		}
		
		
		static handle cast(DynamicValue::Reader src, return_value_policy policy, handle parent) {
			switch(src.getType()) {
				case DynamicValue::VOID: return py::none();
				case DynamicValue::BOOL: return py::cast(src.as<bool>());
				case DynamicValue::INT: return py::cast(src.as<signed long long>());
				case DynamicValue::UINT: return py::cast(src.as<unsigned long long>());
				case DynamicValue::FLOAT: return py::cast(src.as<double>());
				case DynamicValue::TEXT: return py::cast(src.as<capnp::Text>());
				case DynamicValue::DATA: return py::cast(src.as<capnp::Data>());
				case DynamicValue::LIST: return py::cast(src.as<capnp::DynamicList>());
				case DynamicValue::ENUM: return py::cast(src.as<capnp::DynamicEnum>());
				// case DynamicValue::STRUCT: return py::cast(src.as<capnp::DynamicStruct>());
				case DynamicValue::CAPABILITY: return py::cast(src.as<capnp::DynamicCapability>());
				case DynamicValue::ANY_POINTER: return py::cast(src.as<capnp::AnyPointer>());
			}
			
			KJ_REQUIRE(src.getType() == DynamicValue::STRUCT);
			
			DynamicStruct::Reader dynamicStruct = src.as<DynamicStruct>();
			auto typeId = dynamicStruct.getSchema().getProto().getId();
			
			if(globalReaderClasses.contains(py::cast(typeId))) {
				// Retrieve target class to use
				auto targetClass = globalReaderClasses[py::cast(typeId)];
				
				// Construct instance of registered class by calling the class
				// The target class inherits from DynamicStruct::Reader, which has a
				// copy initializer. The target object will be forwarded to it
				auto result = targetClass(dynamicStruct);
				return result;
			}
			
			// TODO: Just default-construct the target class on-demand?
			// In principle, this should never be reached anyway
			return type_caster_base<DynamicStruct::Reader>::cast((DynamicStruct::Reader &&) src, policy, parent);
		}
	};
}}

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