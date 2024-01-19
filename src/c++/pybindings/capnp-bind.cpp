// This translation module is responsible for calling import_array in numpy
#define FSCPY_IMPORT_ARRAY
#include "tensor.h"

#include "fscpy.h"
#include "data.h"
#include "loader.h"
#include "pickle.h"
#include "assign.h"

#include <fsc/common.h>

#include <pybind11/operators.h>

using capnp::AnyPointer;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicCapability;
using capnp::DynamicEnum;
using capnp::DynamicList;

using O = fscpy::CapnpObject;
using R = fscpy::CapnpReader;
using B = fscpy::CapnpBuilder;

namespace fscpy {

static py::module_ capnpModule;

template<typename T, typename... Params>
struct ClassBinding : public py::class_<T, Params...> {
	template<typename... Args>
	ClassBinding(Args&&... args) :
		py::class_<T, Params...>(capnpModule, fwd<Args>(args)...)
	{}
	
	// Gets called by constructor, so return type can be void
	ClassBinding& withCommon(kj::StringPtr cloneName = "clone") {
		this -> def(cloneName.cStr(), &T::clone);
		this -> def("__copy__", &T::clone);
		this -> def("__deepcopy__", [](T& t, py::object memo) { return t.clone(); });
		
		this -> def("__repr__", &T::repr);
		return *this;
	}
	
	template<typename T2 = T>
	ClassBinding& withBuffer() {
		this -> def_buffer(&T2::buffer);
		return *this;
	}
	
	template<typename T2 = T>
	ClassBinding& withGetitem() {
		this -> def("__getitem__", &T2::get);
		return *this;
	}
	
	template<typename T2 = T>
	ClassBinding& withSetitem() {
		this -> def("__setitem__", &T2::get);
		this -> def("__delitem__", &T2::init);
		return *this;
	}
	
	template<typename T2 = T>
	ClassBinding& withSequence() {		
		this -> def("__len__", &T2::size);
		this -> def("__iter__", [](T2& t) {
			return py::make_iterator(t.begin(), t.end());
		}, py::keep_alive<0, 1>());
		return *this;
	}
	
	template<typename T2 = T>
	ClassBinding& withListInterface() {
		this -> withCommon();
		this -> withBuffer<T2>();
		this -> withGetitem<T2>();
		this -> withSequence<T2>();
		
		return *this;
	}
};

void bindRootClasses() {
	ClassBinding<O>("Object")
		.def_property_readonly("type_", &O::getType)
	;
	
	ClassBinding<R>("Reader");
	ClassBinding<B>("Builder");
}

void bindBlobClasses() {
	ClassBinding<DataCommon, O>("Data");
	ClassBinding<DataReader, DataCommon, R>("DataReader", py::buffer_protocol())
		.withCommon()
		.withBuffer()
	;
	ClassBinding<DataBuilder, DataCommon, B>("DataBuilder", py::buffer_protocol())
		.withCommon()
		.withBuffer()
	;
	
	ClassBinding<TextCommon, O>("Text");
	ClassBinding<TextReader, TextCommon, R>("TextReader")
		.withCommon()
	;
	ClassBinding<TextBuilder, TextCommon, B>("TextBuilder")
		.withCommon()
	;
	
	py::implicitly_convertible<DataBuilder, DataReader>();
	py::implicitly_convertible<TextBuilder, TextReader>();
}

void bindAnyClasses() {
	using AR = AnyReader;
	using AB = AnyBuilder;
	
	ClassBinding<AnyCommon>("AnyPointer");
	ClassBinding<AR, AnyCommon, R>("AnyReader")
		.withCommon()
		.def("interpretAs", &AR::interpretAs)
	;
	
	ClassBinding<AB, AnyCommon, B>("AnyBuilder")
		.withCommon()
		.def("set", &AB::setList)
		.def("set", &AB::setStruct)
		.def("set", &AB::setCap)
		.def("set", &AB::adopt)
		.def("interpretAs", &AB::interpretAs)
		.def("initAs", &AB::initBuilderAs, py::arg("type"), py::arg("size") = 0)
		.def("setAs", &AB::assignAs)
	;
	
	py::implicitly_convertible<AB, AR>();
}

void bindListClasses() {
	using L = DynamicListCommon;
	using LB = DynamicListBuilder;
	using LR = DynamicListReader;
	
	ClassBinding<L, O>("List");
	ClassBinding<LB, L, B>("ListBuilder", py::buffer_protocol())
		.withListInterface()
		.def("init", &LB::initList)
	;
	
	ClassBinding<LR, L, R>("ListReader", py::buffer_protocol())
		.withListInterface()
	;
	
	py::implicitly_convertible<LB, LR>();
}

template<typename T, typename... Params>
void bindPipelineCompatibleInterface(ClassBinding<T, Params...>& cls) {
	cls.withGetitem();
	cls.withSequence();
	cls.withCommon("clone_");
	cls.def(py::init<T&>());
	
	cls.def("which_", &T::whichStr);
}

template<typename T, typename... Params>
void bindStructInterface(ClassBinding<T, Params...>& cls) {
	bindPipelineCompatibleInterface(cls);
	
	cls.withBuffer();
	
	cls.def("has_", &T::has);
	cls.def("toYaml_", &T::toYaml, py::arg("flow"));
	cls.def("toDict_", &T::asDict);
	cls.def("totalBytes_", &T::totalBytes);
	
	cls.def_buffer(&T::buffer);
}

void bindStructClasses() {
	using S  = DynamicStructCommon;
	using SB = DynamicStructBuilder;
	using SR = DynamicStructReader;
	using SP = DynamicStructPipeline;
	
	ClassBinding<S, O>("Struct");
	
	ClassBinding<SR, S, R> reader("StructReader", py::multiple_inheritance(), py::buffer_protocol());
	reader
		.def("__reduce_ex__", &pickleReduceReader)
	;
	ClassBinding<SB, S, B> builder("StructBuilder", py::multiple_inheritance(), py::buffer_protocol());
	builder
		.withSetitem()
		
		.def("init_", &SB::init)
		.def("init_", &SB::initList)
		.def("disown_", &SB::disown)
		
		.def("__reduce_ex__", &pickleReduceBuilder)
	;
	ClassBinding<SP> pipeline("StructPipeline", py::multiple_inheritance());
	
	bindStructInterface(reader);
	bindStructInterface(builder);
	bindPipelineCompatibleInterface(pipeline);
	
	py::implicitly_convertible<SB, SR>();
}

void bindCapClasses() {
	using C = DynamicCapabilityClient;
	using S = DynamicCapabilityServer;
	
	ClassBinding<C, O> client("CapabilityClient", py::multiple_inheritance(), py::metaclass(*baseMetaType));
	ClassBinding<S> server("CapabilityServer", py::multiple_inheritance(), py::metaclass(*baseMetaType));
	
	client
		.def(py::init([](C src) { return src; }))
		.def("__reduce_ex__", &pickleReduceRef)
		.def("__await__", [](C& clt) {
			Promise<py::object> whenReady = clt.whenResolved()
			.then([]() mutable -> py::object { return py::none(); });
			
			return convertToAsyncioFuture(mv(whenReady)).attr("__await__")();
		})
	;
}

void bindFieldDescriptors() {
	using FD = FieldDescriptor;
	
	ClassBinding<FD>("FieldDescriptor")
		.def("__get__", &FD::get1, py::arg("obj"), py::arg("type") = py::none())
		.def("__get__", &FD::get2, py::arg("obj"), py::arg("type") = py::none())
		.def("__get__", &FD::get3, py::arg("obj"), py::arg("type") = py::none())
		.def("__get__", &FD::get4, py::arg("obj"), py::arg("type") = py::none())
		
		.def("__set__", &FD::set)
		.def("__del__", &FD::del)
		
		.def_property_readonly("__doc__", &FD::doc)
	;
}

void bindConstants() {
	using C = ConstantValue;
	
	ClassBinding<C>("Constant")
		.def_property_readonly("value", &C::value)
		.def_property_readonly("type", &C::type)
	;
}

void bindEnumClasses() {
	using EI = EnumInterface;
	
	ClassBinding<EI>("Enum")
		.def("__repr__", &EI::repr)
		.def("__eq__", &EI::eq1, py::is_operator())
		.def("__eq__", &EI::eq2, py::is_operator())
		.def("__eq__", &EI::eq3, py::is_operator())
		.def("__reduce_ex__", &pickleReduceEnum)
		.def_property_readonly("raw", &EI::getRaw)
		.def_property_readonly("type", [](EnumInterface& ei) -> capnp::Type {
			return ei.getSchema();
		})
		.def_static("fromRaw", [](capnp::Type t, uint16_t val) {
			return EnumInterface(t.asEnum(), val);
		})
	;
}

void bindUnpicklers() {
	capnpModule.def("_unpickleReader", &unpickleReader);
	capnpModule.def("_unpickleBuilder", &unpickleBuilder);
	capnpModule.def("_unpickleRef", &unpickleRef);
	capnpModule.def("_unpickleEnum", &unpickleEnum);
}

void bindType() {
	ClassBinding<capnp::Type>("Type")
		.def("toProto", [](capnp::Type& t) {
			auto mb = fsc::heapHeld<capnp::MallocMessageBuilder>(1024);
			
			auto root = mb -> initRoot<capnp::schema::Type>();
			fsc::extractType(t, root);
			
			return fscpy::AnyReader(mb.x(), mb -> getRoot<capnp::AnyPointer>());
		})
		.def_static("fromProto", [](fscpy::AnyReader r) {
			return defaultLoader.capnpLoader.getType(r.getAs<capnp::schema::Type>());
		})
		.def_static("fromProto", [](fscpy::AnyBuilder r) {
			return defaultLoader.capnpLoader.getType(r.getAs<capnp::schema::Type>());
		})
		.def(py::self == py::self)
		.def(py::self != py::self)
		.def("listOf", [](capnp::Type t, unsigned int depth) {
			return t.wrapInList(depth);
		}, py::arg("depth") = 1)
	;
}

/*void bindAssignable() {
	ClassBinding<Assignable>("Assignable");
}*/

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
	
	// Static global defined above
	capnpModule = m.def_submodule("capnp", "Python bindings for Cap'n'proto classes (excluding KJ library)");
	
	#define CHECK() if(PyErr_Occurred()) throw py::error_already_set();
	
	bindRootClasses();
	bindListClasses();
	bindBlobClasses();
	bindStructClasses();
	bindFieldDescriptors();
	bindCapClasses();
	bindEnumClasses();
	bindAnyClasses();
	bindType();
	bindConstants();
	bindUnpicklers();
	//bindAssignable();
	
	capnpModule = py::module();
	
	if(PyErr_Occurred())
		throw py::error_already_set();
	// m.def("visualize", &visualizeGraph);
}

}