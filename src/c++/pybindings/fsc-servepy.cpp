#include "servepy.h"

namespace fsc {

namespace {

// The upstream connector will interpret this just as a lvn pointer.
struct ExternLayout {
	fusionsc_LVNHub* lvn = nullptr;
	
	Own<const InProcessServer> server;
	
	ExternLayout(Own<const InProcessServer> srvIn) : server(srvIn -> addRef()) {
		lvn = server -> getHub().release();
	}
	
	~ExternLayout() {
		lvn -> decRef(lvn);
	}
};

static_assert(std::is_standard_layout<ExternLayout>::value);

void serviceDestructor(void* rawPtr) {
	delete static_cast<ExternLayout*>(rawPtr);
}

}

py::object pybindings::createLocalServer(kj::Function<capnp::Capability::Client()> serviceFactory) {
	auto localModule = py::import_("fusionsc.local");
	
	// Retrieve data store
	py::capsule_ dataStoreCapsule = py::reinterpret_borrow<py__capsule>(localModule.attr("getStore")());
	fsc_DataStore* storePtr = dataStoreCapsule;
	
	// Create new FusionSC instance with the correct data store
	fsc::StartOptions opts;
	opts.dataStore.emplace(storePtr);
	
	auto newLib = fsc::newLibrary(opts);
	
	// Create interface
	ExternLayout* externLayout = new ExternLayout(newInProcessServer(mv(serviceFactory), mv(newLib)));
	
	// Put interface into capsule
	py::capsule interfaceCapsule(externLayout, "fusionsc_LvnHub*", &serviceDestructor);
	
	// Use upstream library to connect to capsule
	return localModule.attr("LocalServer")(mv(interfaceCapsule));
}

}