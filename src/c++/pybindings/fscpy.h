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
		inline virtual ~UnknownObject() {};
	};

	template<typename T>
	struct UnknownHolder : public UnknownObject {
		T val;
		UnknownHolder(T t) : val(mv(t)) {}
		~UnknownHolder () noexcept {}
	};
	
	template<typename T>
	UnknownObject* eraseType(T t) { return new UnknownHolder<T>(mv(t)); }
	
	template<typename T>
	py::object unknownObject(T ref) {
		auto holder = new UnknownHolder(mv(ref));
		return py::cast((fscpy::UnknownObject*) holder);
	}		
	
	struct ContiguousCArray {
		kj::Array<unsigned char> data;
		
		std::vector<py::ssize_t> shape;
		size_t elementSize = 0;
		kj::String format;
		
		template<typename T, typename ShapeContainer>
		ArrayPtr<T> alloc(ShapeContainer& requestedShape) {
			size_t shapeProd = 1;
			
			shape.resize(requestedShape.size());
			for(auto i : kj::indices(requestedShape)) {
				shape[i] = requestedShape[i];
				shapeProd *= shape[i];
			}
			
			elementSize = sizeof(T);
			
			data = kj::heapArray<unsigned char>(shapeProd * elementSize);
			return kj::ArrayPtr<T>((T*) data.begin(), shapeProd);
		}
		
		py::buffer_info getBuffer();
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
	
	// Defined in service.cpp
	void loadDefaultSchema(py::module_& m);
	
	// Defined in devices.cpp
	void loadDeviceSchema(py::module_& m);
	
	// constexpr inline kj::StringPtr INTERNAL_ACCESS_KEY = "9821736419873251235"_kj;
	
	py::object methodDescriptor(py::object method);
	py::object simpleObject();
}

extern kj::Own<py::dict> globalClasses;
extern kj::Own<py::type> baseType;
extern kj::Own<py::type> baseMetaType;

// Custom type casters
#include "typecast_kj.h"
#include "typecast_capnp.h"