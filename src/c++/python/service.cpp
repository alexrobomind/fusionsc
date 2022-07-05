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

}

namespace fscpy {
	void initService(py::module_& m) {	
		m.def("connectSameThread", &connectSameThread1);
		m.def("connectSameThread", &connectSameThread2);
		m.def("connectSameThread", &connectRemote1, py::arg("address"), py::arg("port") = 0);
	}

	void loadDefaultSchema(py::module_& m) {
		defaultLoader.addBuiltin<ToroidalGrid, MagneticField, RootService>();
		auto schemas = getBuiltinSchemas<RootService>();
			
		for(auto node : schemas) {
			defaultLoader.importNodeIfRoot(node.getId(), m);
		}
	}
}