#include "fsc-servepy.h"

#include <fsc/capi-lvn.h>
#include <fsc/capi-store.h>
#include <fsc/services.h>
#include <fsc/local.h>

namespace py = pybind11;

using kj::Own;

namespace fsc {

namespace {

// The upstream connector will interpret this just as a lvn pointer.
struct ExternLayout {
	fusionsc_LvnHub* lvn = nullptr;
	
	Own<const InProcessServer> server;
	
	ExternLayout(Own<const InProcessServer> srvIn) : server(srvIn -> addRef()) {
		lvn = server -> getHub().release();
	}
	
	~ExternLayout() {
		lvn -> decRef(lvn);
	}
};

static_assert(std::is_standard_layout<ExternLayout>::value);

void serviceDestructor(PyObject* objectPtr) {
	void* contentPtr = PyCapsule_GetPointer(objectPtr, "fusionsc_LvnHub*");
	delete static_cast<ExternLayout*>(contentPtr);
}

}

Library pybindings::newPythonBoundLibrary() {
	auto localModule = py::module_::import("fusionsc.local");
	
	// Retrieve data store
	py::capsule dataStoreCapsule = py::reinterpret_borrow<py::capsule>(localModule.attr("getStore")());
	fusionsc_DataStore* storePtr = dataStoreCapsule;
	
	// Create new FusionSC instance with the correct data store
	fsc::StartupParameters opts;
	opts.dataStore.emplace(storePtr);
	
	return fsc::newLibrary(opts);
}

py::object pybindings::createLocalServer(kj::Function<capnp::Capability::Client()> serviceFactory, capnp::InterfaceSchema schema) {
	KJ_REQUIRE(!schema.isBranded(), "Only unbranded schemas can currently be served as local servers. This restriction might be relaxed in future.");
	
	// Create interface
	ExternLayout* externLayout = new ExternLayout(newInProcessServer(mv(serviceFactory), newPythonBoundLibrary()));
	
	// Put interface into capsule
	py::capsule interfaceCapsule(externLayout, "fusionsc_LvnHub*", &serviceDestructor);
	
	// Get server type
	auto loaderModule = py::module_::import("fusionsc").attr("loader");
	auto pyType = loaderModule.attr("getType")(schema.getProto().getId());
	py::print("Server type: ", pyType);
	
	// Use upstream library to connect to capsule
	auto localModule = py::module_::import("fusionsc.local");
	return localModule.attr("LocalServer")(mv(interfaceCapsule), mv(pyType));
}

}
