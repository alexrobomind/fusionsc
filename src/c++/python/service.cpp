#include "fscpy.h"
#include "async.h"
#include "loader.h"
#include "typecast_capnp.h"

#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/schema-loader.h>

#include <kj/string-tree.h>

#include <pybind11/pybind11.h>

#include <fsc/services.h>
#include <fsc/magnetics.h>
#include <fsc/offline.capnp.h>

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

using capnp::Schema;
using capnp::StructSchema;

using namespace fscpy;

namespace {

RootService::Client connectSameThread1(RootConfig::Reader config) {
	fscpy::PyContext::startEventLoop();
	return createRoot(config);
}
	
RootService::Client connectSameThread2() {
	Temporary<RootConfig> config;
	return connectSameThread1(config);
}

RootService::Client connectRemote1(kj::StringPtr address, unsigned int port) {
	fscpy::PyContext::startEventLoop();
	KJ_UNIMPLEMENTED("Remote connection not implemented");
	return connectRemote(address, port);
}

RootService::Client connectLocal1() {
	fscpy::PyContext::startEventLoop();
	auto serverFactory = newInProcessServer<RootService>([]() mutable {
		Temporary<RootConfig> rootConfig;
		return createRoot(rootConfig);
	});
	
	auto server = serverFactory();
	return attach(server, mv(serverFactory));
}

ResolverChain::Client newResolverChain1() {
	fscpy::PyContext::startEventLoop();
	return newResolverChain();
}

}

namespace fscpy {
	void initService(py::module_& m) {	
		m.def("connectSameThread", &connectSameThread1);
		m.def("connectSameThread", &connectSameThread2);
		m.def("connectLocal", &connectLocal1);
		m.def("connectSameThread", &connectRemote1, py::arg("address"), py::arg("port") = 0);
		m.def("newResolverChain", &newResolverChain1);
		m.def("newCache", &fsc::newCache);
	}

	void loadDefaultSchema(py::module_& m) {
		defaultLoader.addBuiltin<
			//ToroidalGrid,
			//MagneticField,
			RootService,
			OfflineData,
			ResolverChain
		>();
		
		auto schemas = getBuiltinSchemas<FieldResolver, GeometryResolver, RootService, OfflineData, ResolverChain>();
			
		for(auto node : schemas) {
			defaultLoader.importNodeIfRoot(node.getId(), m);
		}
	}
}