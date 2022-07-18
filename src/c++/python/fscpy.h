#pragma once

#include <pybind11/pybind11.h>
#include <fsc/common.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, kj::Own<T>);

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
	
	// Init methods for various components
	void initAsync(py::module_& m);
	void initCapnp(py::module_& m);
	void initData(py::module_& m);
	void initDevices(py::module_& m);
	void initKj(py::module_& m);
	void initLoader(py::module_& m);	
	void initService(py::module_& m);
	
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