// This translation module is responsible for calling import_array in numpy
#define FSCPY_IMPORT_ARRAY

#include "tensor.h"

#include "fscpy.h"
#include "data.h"
#include "loader.h"

using capnp::AnyPointer;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicCapability;
using capnp::DynamicEnum;
using capnp::DynamicList;

namespace fscpy {

Maybe<DynamicValue::Reader> dynamicValueFromScalar(py::handle handle) {
	// 0D arrays
	if(PyArray_IsZeroDim(handle.ptr())) {
		PyArrayObject* scalarPtr = reinterpret_cast<PyArrayObject*>(handle.ptr());
		
		switch(PyArray_TYPE(scalarPtr)) {
			#define HANDLE_NPY_TYPE(npytype, ctype) \
				case npytype: { \
					ctype* data = static_cast<ctype*>(PyArray_DATA(scalarPtr)); \
					return DynamicValue::Reader(*data); \
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
				return DynamicValue::Reader((*data) != 0);
			}
				
			default:
				break;
		}
	}
	
	// NumPy scalars
	if(PyArray_IsScalar(handle.ptr(), Bool)) { \
		return DynamicValue::Reader(PyArrayScalar_VAL(handle.ptr(), Bool) != 0); \
	}
	
	#define HANDLE_TYPE(cls) \
		if(PyArray_IsScalar(handle.ptr(), cls)) { \
			return DynamicValue::Reader(PyArrayScalar_VAL(handle.ptr(), cls)); \
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
			return DynamicValue::Reader(cTyped); \
		}
		
	// Bool is a subtype of int, so this has to go first
	HANDLE_TYPE(bool, py::bool_);
	HANDLE_TYPE(signed long long, py::int_);
	HANDLE_TYPE(double, py::float_);
	
	#undef HANDLE_TYPE
	
	return nullptr;
}


// Pickling support

kj::Array<const byte> fromPythonBuffer(py::buffer buf) {
	auto bufInfo = buf.request();
	
	kj::ArrayPtr<const byte> ptr((const byte*) bufInfo.ptr, bufInfo.itemsize * bufInfo.size);
	
	// Add a deleter for the buffer info that deletes inside the GIL
	Maybe<py::buffer_info> deletable = mv(bufInfo);
	return ptr.attach(kj::defer([deletable = mv(deletable)]() mutable {
		py::gil_scoped_acquire withGil;
		deletable = nullptr;
	}));
}

py::list flattenDataRef(uint32_t pickleVersion, capnp::DynamicCapability::Client dynamicRef) {
	auto payloadType = getRefPayload(dynamicRef.getSchema());
	
	auto data = PythonWaitScope::wait(getActiveThread().dataService().downloadFlat(dynamicRef.castAs<DataRef<>>()));
	
	py::list result(data.size());
	
	if(pickleVersion <= 4) {
		// Version with copying
		for(auto i : kj::indices(data)) {
			result[i] = py::bytes((const char*) data[i].begin(), (uint64_t) data[i].size());
		}
	} else {
		// Zero-copy version
		auto pbCls = py::module_::import("pickle").attr("PickleBuffer");
		for(auto i : kj::indices(data)) {
			py::object asPy = py::cast(capnp::Data::Reader(data[i].asPtr()));
			asPy.attr("_backingArray") = unknownObject(mv(data[i]));
			
			result[i] = pbCls(mv(asPy));
		}
	}
	
	return result;
}

LocalDataRef<> unflattenDataRef(py::list input) {
	auto arrayBuilder = kj::heapArrayBuilder<kj::Array<const byte>>(input.size());
	
	for(auto i : kj::indices(input)) {
		arrayBuilder.add(fromPythonBuffer(py::reinterpret_borrow<py::buffer>(input[i])));
	}
	
	return getActiveThread().dataService().publishFlat<capnp::AnyPointer>(arrayBuilder.finish());
}

void bindPickleRef(py::module_& m, py::class_<capnp::DynamicCapability::Client> cls) {
	// Note: pybind11 generates incorrect qualnames for functions
	// Therefore, we need to define unpicklers and then call them via a
	// wrapper defined by python itself.
	
	m.def("_unpickleRef", [](uint32_t version, py::list data) -> capnp::DynamicCapability::Client {
		KJ_REQUIRE(version == 1, "Only version 1 representation supported");
		return unflattenDataRef(data);
	});
	
	cls.def("__reduce_ex__", [cls](capnp::DynamicCapability::Client src, uint32_t pickleVersion) {
		auto unpickler = py::module_::import("fusionsc").attr("pickle_support").attr("unpickleRef");
		return py::make_tuple(
			unpickler,
			py::make_tuple(
				1,
				flattenDataRef(pickleVersion, src)
			)
		);
	});
}

void bindPickleReader(py::module_& m, py::class_<DynamicStructReader> cls) {
	// Note: pybind11 generates incorrect qualnames for functions
	// Therefore, we need to define unpicklers and then call them via a
	// wrapper defined by python itself.
	
	m.def("_unpickleReader", [](uint32_t version, py::list data) {
		KJ_REQUIRE(version == 1, "Only version 1 representation supported");
		auto ref = unflattenDataRef(data);
		return openRef(capnp::schema::Type::AnyPointer::Unconstrained::STRUCT, mv(ref));
	});
	
	cls.def("__reduce_ex__", [cls](capnp::DynamicStruct::Reader src, uint32_t pickleVersion) mutable {
		auto unpickler = py::module_::import("fusionsc").attr("pickle_support").attr("unpickleReader");
		return py::make_tuple(
			unpickler,
			py::make_tuple(
				1,
				flattenDataRef(pickleVersion, publishReader(src))
			)
		);
	});
}

void bindPickleBuilder(py::module_& m, py::class_<DynamicStructBuilder> cls) {
	// Note: pybind11 generates incorrect qualnames for functions
	// Therefore, we need to define unpicklers and then call them via a
	// wrapper defined by python itself.
	
	m.def("_unpickleBuilder", [](uint32_t version, py::list data) {
		KJ_REQUIRE(version == 1, "Only version 1 representation supported");
		auto ref = unflattenDataRef(data);
		
		KJ_IF_MAYBE(pPayloadType, getPayloadType(ref)) {
			auto msgBuilder = kj::heap<capnp::MallocMessageBuilder>();
			msgBuilder -> setRoot(ref.get());
			
			capnp::DynamicStruct::Builder dynamic = msgBuilder -> getRoot<capnp::DynamicStruct>(pPayloadType -> asStruct());
			py::object result = py::cast(dynamic);
			
			result.attr("_msg") = unknownObject(mv(msgBuilder));
			
			return result;
		} else {
			KJ_FAIL_REQUIRE("Payload type missing, can't unpickle");
		}
	});
	
	cls.def("__reduce_ex__", [cls](capnp::DynamicStruct::Builder src, uint32_t pickleVersion) mutable {
		auto unpickler = py::module_::import("fusionsc").attr("pickle_support").attr("unpickleBuilder");
		return py::make_tuple(
			unpickler,
			py::make_tuple(
				1,
				flattenDataRef(pickleVersion, publishBuilder(src))
			)
		);
	});
}

void bindBlobClasses(py::module_& m) {
	using DR = DataReader;
	using DB = DataBuilder;
	using TR = TextReader;
	// TB is alias for TR
	
	py::class_<DR>(m, "DataReader")
		.def("__repr__", &DR::repr)
		.def_buffer(&DR::buffer)
	;
	
	py::class_<DB>(m, "DataBuilder")
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
	
	py::class_<LB> builder(m, "ListBuilder");
	builder
		.def("__setitem__", &LB::set)
		.def("init", &LB::initList)
	;
	
	py::class_<LR> reader(m, "ListReader");
	
	bindListInterface(builder);
	bindListInterface(reader);
	
	py::implicitly_convertible<LB, LR>();
}

template<typename T, typename... Params>
void bindPipelineCompatibleInterface(py::class_<T, Params...>& cls) {
	cls.def("__getitem__", &T::get);
	cls.def("which_", &T::whichStr);
	cls.def("__repr__", &T::repr);
}

template<typename T, typename... Params>
void bindStructInterface(py::class_<T, Params...>& cls) {
	bindPipelineCompatibleInterface(cls);
	cls.def("has_", &T::has);
	cls.def("toYaml_", &T::toYaml, py::arg("flow"));
	cls.def("__len__", &T::size);
	cls.def("toDict_", &T::asDict);
	cls.def("clone_", &T::clone);
	cls.def_buffer(&T::buffer);
}

void bindStructClasses(py::module_& m) {
	using SB = DynamicStructBuilder;
	using SR = DynamicStructReader;
	using SP = DynamicStructPipeline;
	
	py::class_<SR> reader(m, "StructReader", py::multiple_inheritance());
	py::class_<SB> builder(m, "StructBuilder", py::multiple_inheritance());
	builder
		.def("__setitem__", &SB::set)
		.def("init_", &SB::initList)
		.def("disown_", &SB::disown)
	;
	py::class_<SP> pipeline(m, "StructPipeline", py::multiple_inheritance());
	
	bindStructInterface(reader);
	bindStructInterface(builder);
	bindPipelineCompatibleInterface(pipeline);
	
	bindPickleReader(m, reader);
	bindPickleBuilder(m, builder);
	
	py::implicitly_convertible<SB, SR>();
}

void bindCapClasses(py::module_& m) {
	using C = DynamicCapabilityClient;
	using S = DynamicCapabilityServer;
	
	py::class_<C> client(m, "CapabilityClient", py::multiple_inheritance(), py::metaclass(*baseMetaType));
	py::class_<S> server(m, "CapabilityServer", py::multiple_inheritance(), py::metaclass(*baseMetaType));
	
	client.def(py::init([](C src) { return src; }));
	
	bindPickleRef(m, client);
}

void bindFieldDescriptors(py::module& m) {
	using FD = FieldDescriptor;
	
	py::class_<FD>(m, "FieldDescriptor")
		.def("__get__", &FD::get1)
		.def("__get__", &FD::get2)
		.def("__get__", &FD::get3)
		
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
	
	bindListClasses(mcapnp);
	bindBlobClasses(mcapnp);
	bindStructClasses(mcapnp);
	bindFieldDescriptors(mcapnp);
	bindCapClasses(mcapnp);
	bindEnumClasses(mcapnp);
	bindHelpers(mcapnp);
	bindAnyClasses(mcapnp);
	
	m.add_object("void", py::cast(capnp::DynamicValue::Reader(capnp::Void())));
	
	m.def("visualize", &visualizeGraph);
}

}