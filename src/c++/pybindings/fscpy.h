#pragma once

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <fsc/common.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, kj::Own<T>);

#define FSC_NATIVE_MODULE "fusionsc.native"

namespace py = pybind11;

namespace fscpy {
	using namespace fsc;

	struct UnknownObject {
		const int* type = nullptr;
		inline virtual ~UnknownObject() {};
	};
	
	template<typename T>
	UnknownObject* eraseType(T&& t);
	
	template<typename T>
	Maybe<T&> checkType(UnknownObject& o);
	
	template<typename T>
	py::object unknownObject(T&& ref);
		
	struct ContiguousCArray {
		kj::Array<unsigned char> data;
		
		std::vector<py::ssize_t> shape;
		size_t elementSize = 0;
		kj::String format;
		
		py::buffer_info getBuffer();
		
		template<typename T, typename ShapeContainer>
		static ContiguousCArray alloc(ShapeContainer& requestedShape, kj::StringPtr formatCode);
		
		template<typename T>
		kj::ArrayPtr<T> as();
	};
	
	// Init methods for various components
	void initAsync(py::module_& m);
	void initCapnp(py::module_& m);
	void initData(py::module_& m);
	void initDevices(py::module_& m);
	void initKj(py::module_& m);
	void initLoader(py::module_& m);	
	void initService(py::module_& m);
	void initHelpers(py::module_& m);
	void initStructio(py::module_& m);
		
	py::object methodDescriptor(py::object method);
}

#include "fscpy-inl.h"

#include "kj.h"

#define FSCPY_MOVE_ONLY_CASTER \
	type_caster() = default; \
	type_caster(const type_caster& other) = delete; \
	type_caster(type_caster&& other) = default; \

// Custom type casters
#include "typecast_kj.h"
#include "typecast_capnp.h"
