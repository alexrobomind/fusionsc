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
	
	HANDLE_TYPE(Byte);
	HANDLE_TYPE(Short);
	HANDLE_TYPE(Int);
	HANDLE_TYPE(Long);
	HANDLE_TYPE(LongLong);
	
	HANDLE_TYPE(UByte);
	HANDLE_TYPE(UShort);
	HANDLE_TYPE(UInt);
	HANDLE_TYPE(ULong);
	HANDLE_TYPE(ULongLong);
	
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

// Pickling support

void bindBlobClasses(py::module_& m) {
	using DR = DataReader;
	using DB = DataBuilder;
	using TR = TextReader;
	// TB is alias for TR
	
	py::class_<DR>(m, "DataReader", py::buffer_protocol())
		.def("__repr__", &DR::repr)
		.def_buffer(&DR::buffer)
	;
	
	py::class_<DB>(m, "DataBuilder", py::buffer_protocol())
		.def("__repr__", &DB::repr)
		.def_buffer(&DB::buffer)
	;
	
	py::class_<TR>(m, "TextReader")
		.def("__repr__", &TR::repr)
	;
	
	py::implicitly_convertible<DB, DR>();
}

void bindAnyClasses(py::module& m) {
	using AR = AnyReader;
	using AB = AnyBuilder;
	
	py::class_<AR>(m, "AnyReader")
		.def("__repr__", &AR::repr)
	;
	
	py::class_<AB>(m, "AnyBuilder")
		.def("__repr__", &AB::repr)
		.def("set", &AB::setList)
		.def("set", &AB::setStruct)
		.def("set", &AB::adopt)
	;
	
	py::implicitly_convertible<AB, AR>();
}

template<typename T, typename... Params>
void bindListInterface(py::class_<T, Params...>& cls) {
	cls.def("__getitem__", &T::get);
	cls.def("clone", &T::clone);
	cls.def("__len__", &T::size);
	cls.def_buffer(&T::buffer);
	
	cls.def(
		"__iter__",
		[](T& t) { return py::make_iterator(t.begin(), t.end()); },
		py::keep_alive<0, 1>()
	);
}

void bindListClasses(py::module_& m) {
	using LB = DynamicListBuilder;
	using LR = DynamicListReader;
	
	py::class_<LB> builder(m, "ListBuilder", py::buffer_protocol());
	builder
		.def("__setitem__", &LB::set)
		.def("init", &LB::initList)
	;
	
	py::class_<LR> reader(m, "ListReader", py::buffer_protocol());
	
	bindListInterface(builder);
	bindListInterface(reader);
	
	py::implicitly_convertible<LB, LR>();
}

template<typename T, typename... Params>
void bindPipelineCompatibleInterface(py::class_<T, Params...>& cls) {
	cls.def("__getitem__", &T::get);
	cls.def("which_", &T::whichStr);
	cls.def("__repr__", &T::repr);
	cls.def(py::init<T&>());
}

template<typename T, typename... Params>
void bindStructInterface(py::class_<T, Params...>& cls) {
	bindPipelineCompatibleInterface(cls);
	cls.def("has_", &T::has);
	cls.def("toYaml_", &T::toYaml, py::arg("flow"));
	cls.def("__len__", &T::size);
	cls.def("toDict_", &T::asDict);
	cls.def("clone_", &T::clone);
	cls.def("totalBytes", &T::totalBytes);
	cls.def_buffer(&T::buffer);
}

void bindStructClasses(py::module_& m) {
	using SB = DynamicStructBuilder;
	using SR = DynamicStructReader;
	using SP = DynamicStructPipeline;
	
	py::class_<SR> reader(m, "StructReader", py::multiple_inheritance(), py::buffer_protocol());
	reader
		.def("__reduce_ex__", &pickleReduceReader)
	;
	py::class_<SB> builder(m, "StructBuilder", py::multiple_inheritance(), py::buffer_protocol());
	builder
		.def("__setitem__", &SB::set)
		.def("init_", &SB::init)
		.def("init_", &SB::initList)
		.def("disown_", &SB::disown)
		.def("__reduce_ex__", &pickleReduceBuilder)
	;
	py::class_<SP> pipeline(m, "StructPipeline", py::multiple_inheritance());
	
	bindStructInterface(reader);
	bindStructInterface(builder);
	bindPipelineCompatibleInterface(pipeline);
	
	py::implicitly_convertible<SB, SR>();
}

void bindCapClasses(py::module_& m) {
	using C = DynamicCapabilityClient;
	using S = DynamicCapabilityServer;
	
	py::class_<C> client(m, "CapabilityClient", py::multiple_inheritance(), py::metaclass(*baseMetaType));
	py::class_<S> server(m, "CapabilityServer", py::multiple_inheritance(), py::metaclass(*baseMetaType));
	
	client
		.def(py::init([](C src) { return src; }))
		.def("__reduce_ex__", &pickleReduceRef)
	;
}

void bindFieldDescriptors(py::module& m) {
	using FD = FieldDescriptor;
	
	py::class_<FD>(m, "FieldDescriptor")
		.def("__get__", &FD::get1, py::arg("obj"), py::arg("type") = py::none())
		.def("__get__", &FD::get2, py::arg("obj"), py::arg("type") = py::none())
		.def("__get__", &FD::get3, py::arg("obj"), py::arg("type") = py::none())
		.def("__get__", &FD::get4, py::arg("obj"), py::arg("type") = py::none())
		
		.def("__set__", &FD::set)
		.def("__del__", &FD::del)
	;
}


void bindEnumClasses(py::module_& m) {
	using EI = EnumInterface;
	
	py::class_<EI>(m, "Enum")
		.def("__repr__", &EI::repr)
		.def("__eq__", &EI::eq1, py::is_operator())
		.def("__eq__", &EI::eq2, py::is_operator())
		.def("__eq__", &EI::eq3, py::is_operator())
	;
}

void bindUnpicklers(py::module_& m) {
	m.def("_unpickleReader", &unpickleReader);
	m.def("_unpickleBuilder", &unpickleBuilder);
	m.def("_unpickleRef", &unpickleRef);
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
	
	py::module_ mcapnp = m.def_submodule("capnp", "Python bindings for Cap'n'proto classes (excluding KJ library)");
	
	#define CHECK() if(PyErr_Occurred()) throw py::error_already_set();
	
	bindListClasses(mcapnp);
	bindBlobClasses(mcapnp);
	bindStructClasses(mcapnp);
	bindFieldDescriptors(mcapnp);
	bindCapClasses(mcapnp);
	bindEnumClasses(mcapnp);
	bindAnyClasses(mcapnp);
	bindUnpicklers(mcapnp);
	
	if(PyErr_Occurred())
		throw py::error_already_set();
	// m.def("visualize", &visualizeGraph);
}

}