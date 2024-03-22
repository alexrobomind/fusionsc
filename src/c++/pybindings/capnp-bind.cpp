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
#include <pybind11/options.h>

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
	
namespace {

py::module_ capnpModule;

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
	using CC = DynamicCallContext;
	
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
		.def_property_readonly("executor_", &C::executor)
	;
	
	server
		.def(py::init<capnp::InterfaceSchema>())
		.def("asClient", &S::thisCap)
	;
	
	ClassBinding<CC>("CallContext")
		.def_property_readonly("params", &CC::getParams)
		.def_property("results", &CC::getResults, &CC::setResults)
		.def("initResults", &CC::initResults)
		.def("tailCall", &CC::tailCall)
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

struct MethodDict {
	capnp::InterfaceSchema schema;
	MethodDict(capnp::InterfaceSchema s) : schema(s) {}
	
	MethodInfo getItem(kj::StringPtr key) {
		return MethodInfo(schema.getMethodByName(key));
	}
};

kj::String asRepr(kj::StringPtr moduleName, kj::StringTree qualName) {
	return kj::str(moduleName, ".", mv(qualName));
}

py::type builderFor(capnp::Type type) {
	return defaultLoader.builderType(type);
}

py::type readerFor(capnp::Type type) {
	return defaultLoader.readerType(type);
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
		.def("interpret", [](capnp::Type& t) -> capnp::Type {
			return t;
		})
		.def("__repr__", [](capnp::Type& t) -> kj::String {
			return kj::apply(asRepr, defaultLoader.qualName(t));
		})
		.def_property_readonly("Builder", &builderFor)
		.def_property_readonly("Reader", &readerFor)
	;
	
	ClassBinding<capnp::Schema> schema("Schema");
	schema.def("__repr__", [](capnp::Schema& s) -> kj::String {
		return kj::apply(asRepr, defaultLoader.qualName(s));
	});
	schema.def("__getattr__", [](capnp::Schema& self, kj::StringPtr name) -> capnp::Type {
		using Brand = capnp::schema::Brand;
		auto proto = self.getProto();
		
		// Struct nodes can have groups, that also appears as nested nodes
		if(proto.isStruct()) {
			auto asStruct = self.asStruct();
	
			for(capnp::StructSchema::Field field : asStruct.getFields()) {
				if(!field.getProto().isGroup())
					continue;
				
				if(field.getProto().getName() != name)
					continue;
				
				return field.getType();
			}
		}
		
		// Look up child ID
		for(auto nestedNode : proto.getNestedNodes()) {
			if(nestedNode.getName() != name)
				continue;
			
			// Load type from default loader
			defaultLoader.capnpLoader.get(nestedNode.getId(), Brand::Reader(), self);
		}
		
		throw py::attribute_error();
	});
	schema.def("__dir__", [](capnp::Schema& self) {
		py::list result;
		result.append("__getitem__");
		
		auto proto = self.getProto();
		
		if(proto.isStruct()) {
			auto asStruct = self.asStruct();
			result.append("Builder");
			result.append("Reader");
			result.append("Pipeline");
			result.append("newMessage");
	
			for(capnp::StructSchema::Field field : asStruct.getFields()) {
				if(!field.getProto().isGroup())
					continue;
				
				result.append(field.getProto().getName().cStr());
			}
		}
		
		if(proto.isInterface()) {
			auto asInterface = self.asInterface();
			result.append("Client");
			result.append("Server");
			result.append("methods");
		}
		
		// Look up child ID
		for(auto nestedNode : proto.getNestedNodes()) {
			result.append(nestedNode.getName().cStr());
		}
		
		return result;
	});
	schema.def("__getitem__", [](capnp::Schema& self, py::object key) {		
		kj::Vector<Maybe<capnp::Type>> bindings;
		
		if(py::isinstance<py::sequence>(key)) {
			auto seq = py::reinterpret_borrow<py::sequence>(key);
			for(auto binding : seq) {
				if(binding.is_none()) {
					bindings.add(nullptr);
				} else {
					bindings.add(py::cast<capnp::Type>(binding));
				}
			}
		} else {
			if(key.is_none()) {
				bindings.add(nullptr);
			} else {
				py::detail::type_caster<capnp::Type> asType;
				KJ_REQUIRE(asType.load(key, false), "Specialization parameters must be a type or tuple of types");
				bindings.add((capnp::Type) asType);
			}
		}
		
		using Brand = capnp::schema::Brand;
		Temporary<Brand> brand;
		fsc::extractBrand(self, brand);
		
		for(auto scope : brand.getScopes()) {
			if(scope.getScopeId() != self.getProto().getId())
				continue;
			
			auto bindingsOut = scope.initBind(bindings.size());
			for(auto i : kj::indices(bindingsOut)) {
				auto out = bindingsOut[i];
				KJ_IF_MAYBE(pBinding, bindings[i]) {
					fsc::extractType(*pBinding, out.initType());
				} else {
					out.setUnbound();
				}
			}
				
			goto done;
		}
		KJ_FAIL_REQUIRE("The requested type can not be specialized");
		
		done:
		
		return defaultLoader.capnpLoader.get(self.getProto().getId(), brand);
	});
	
	ClassBinding<capnp::StructSchema, capnp::Schema>("StructSchema")
		.def("newMessage", [](capnp::StructSchema& self, py::object copyFrom, size_t initialSize) {
			auto msg = kj::heap<capnp::MallocMessageBuilder>(initialSize);
			
			capnp::DynamicStruct::Builder builder;
			if(copyFrom.is_none()) {
				builder = msg->initRoot<capnp::DynamicStruct>(self);
			} else {
				// Let's try using our assignment logic
				assign(*msg, self, copyFrom);
				builder = msg->getRoot<capnp::DynamicStruct>(self);
			}
			
			return DynamicStructBuilder(mv(msg), builder);
		})
		.def_property_readonly("Builder", &builderFor)
		.def_property_readonly("Reader", &readerFor)
		.def_property_readonly("Pipeline", [](capnp::StructSchema& schema) {
			return defaultLoader.pipelineType(schema);
		})
	;
	
	ClassBinding<capnp::InterfaceSchema, capnp::Schema>("InterfaceSchema")
		.def_property_readonly("methods", [](capnp::InterfaceSchema& self) {
			return MethodDict(self);
		})
		.def_property_readonly("Client", [](capnp::InterfaceSchema& schema) {
			return defaultLoader.clientType(schema);
		})
	;
	
	ClassBinding<capnp::ListSchema>("ListSchema")
		.def_property_readonly("elementType", [](capnp::ListSchema& self) {
			return self.getElementType();
		})
	;
	
	py::implicitly_convertible<capnp::StructSchema, capnp::Type>();
	py::implicitly_convertible<capnp::ListSchema, capnp::Type>();
	py::implicitly_convertible<capnp::InterfaceSchema, capnp::Type>();
}

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
	
	py::options options;
	options.disable_function_signatures();
	
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
