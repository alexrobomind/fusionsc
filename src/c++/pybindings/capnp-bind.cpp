// This translation module is responsible for calling import_array in numpy
#define FSCPY_IMPORT_ARRAY
#include "tensor.h"

#include "fscpy.h"
#include "data.h"
#include "loader.h"
#include "pickle.h"

using capnp::AnyPointer;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicCapability;
using capnp::DynamicEnum;
using capnp::DynamicList;

using O = fscpy::CapnpObject;
using R = fscpy::CapnpReader;
using B = fscpy::CapnpBuilder;

static const int ANONYMOUS = 0;

namespace fscpy {

Maybe<DynamicValueReader> dynamicValueFromScalar(py::handle handle) {
	// 0D arrays
	if(PyArray_IsZeroDim(handle.ptr())) {
		PyArrayObject* scalarPtr = reinterpret_cast<PyArrayObject*>(handle.ptr());
		
		switch(PyArray_TYPE(scalarPtr)) {
			#define HANDLE_NPY_TYPE(npytype, ctype) \
				case npytype: { \
					ctype* data = static_cast<ctype*>(PyArray_DATA(scalarPtr)); \
					return DynamicValueReader(kj::attachRef(ANONYMOUS), *data); \
				}
						
			HANDLE_NPY_TYPE(NPY_INT8,  int8_t);
			HANDLE_NPY_TYPE(NPY_INT16, int16_t);
			HANDLE_NPY_TYPE(NPY_INT32, int32_t);
			HANDLE_NPY_TYPE(NPY_INT64, int64_t);
			
			HANDLE_NPY_TYPE(NPY_UINT8,  uint8_t);
			HANDLE_NPY_TYPE(NPY_UINT16, uint16_t);
			HANDLE_NPY_TYPE(NPY_UINT32, uint32_t);
			HANDLE_NPY_TYPE(NPY_UINT64, uint64_t);
			
			HANDLE_NPY_TYPE(NPY_FLOAT32, float);
			HANDLE_NPY_TYPE(NPY_FLOAT64, double);
			
			#undef HANDLE_NPY_TYPE
			
			case NPY_BOOL: {
				unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(scalarPtr)); 
				return DynamicValueReader(kj::attachRef(ANONYMOUS), (*data) != 0);
			}
				
			default:
				break;
		}
	}
	
	// NumPy scalars
	if(PyArray_IsScalar(handle.ptr(), Bool)) { \
		return DynamicValueReader(kj::attachRef(ANONYMOUS), PyArrayScalar_VAL(handle.ptr(), Bool) != 0); \
	}
	
	#define HANDLE_TYPE(cls) \
		if(PyArray_IsScalar(handle.ptr(), cls)) { \
			return DynamicValueReader(kj::attachRef(ANONYMOUS), PyArrayScalar_VAL(handle.ptr(), cls)); \
		}
	
	HANDLE_TYPE(UByte);
	HANDLE_TYPE(UShort);
	HANDLE_TYPE(UInt);
	HANDLE_TYPE(ULong);
	HANDLE_TYPE(ULongLong);
	
	HANDLE_TYPE(Byte);
	HANDLE_TYPE(Short);
	HANDLE_TYPE(Int);
	HANDLE_TYPE(Long);
	HANDLE_TYPE(LongLong);
	
	HANDLE_TYPE(Float);
	HANDLE_TYPE(Double);
	
	#undef HANDLE_TYPE		
	
	// Python builtins
	#define HANDLE_TYPE(ctype, pytype) \
		if(py::isinstance<pytype>(handle)) { \
			pytype typed = py::reinterpret_borrow<pytype>(handle); \
			ctype cTyped = static_cast<ctype>(typed); \
			return DynamicValueReader(kj::attachRef(ANONYMOUS), cTyped); \
		}
		
	// Bool is a subtype of int, so this has to go first
	HANDLE_TYPE(bool, py::bool_);
	HANDLE_TYPE(signed long long, py::int_);
	HANDLE_TYPE(double, py::float_);
	
	#undef HANDLE_TYPE
	
	return nullptr;
}

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
	ClassBinding<O>("Object");
	ClassBinding<R>("Reader");
	ClassBinding<B>("Builder"):
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
	
	ClassBinding<TextCommon, O>();
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
	;
	
	ClassBinding<AB, AnyCommon, B>("AnyBuilder")
		.withCommon()
		.def("set", &AB::setList)
		.def("set", &AB::setStruct)
		.def("set", &AB::setCap)
		.def("set", &AB::adopt)
	;
	
	py::implicitly_convertible<AB, AR>();
}

void bindListClasses() {
	using L = DynamicListCommon;
	using LB = DynamicListBuilder;
	using LR = DynamicListReader;
	
	ClassBinding<L>("List");
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
	cls.def_property_readonly("schema_", &T::encodeSchema);
	
	cls.def_buffer(&T::buffer);
}

void bindStructClasses() {
	using S  = DynamicStructCommon;
	using SB = DynamicStructBuilder;
	using SR = DynamicStructReader;
	using SP = DynamicStructPipeline;
	
	ClassBinding<S>("Struct");
	
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
	
	ClassBinding<C, P> client("CapabilityClient", py::multiple_inheritance(), py::metaclass(*baseMetaType));
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
	;
}

void bindUnpicklers() {
	capnpModule.def("_unpickleReader", &unpickleReader);
	capnpModule.def("_unpickleBuilder", &unpickleBuilder);
	capnpModule.def("_unpickleRef", &unpickleRef);
	capnpModule.def("_unpickleEnum", &unpickleEnum);
}

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
	
	bindListClasses();
	bindBlobClasses();
	bindStructClasses();
	bindFieldDescriptors();
	bindCapClasses();
	bindEnumClasses();
	bindAnyClasses();
	bindUnpicklers();
	
	capnpModule = py::module();
	
	if(PyErr_Occurred())
		throw py::error_already_set();
	// m.def("visualize", &visualizeGraph);
}

}