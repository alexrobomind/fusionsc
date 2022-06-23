#include "fscpy.h"
#include "async.h"
#include "loader.h"
#include "typecast_capnp.h"

#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/schema-loader.h>

#include <kj/string-tree.h>

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

/*
template<typename... T>
auto getAndAddBuiltin() {
	defaultLoader.addBuiltin<T...>();
	return getBuiltinSchemas<T...>();
}
*/
	
DynamicCapability::Client connectLocal1(RootConfig::Reader config) {
	LibraryThread lt = fscpy::PyContext::libraryThread();
	return createRoot(lt, config);
}
	
DynamicCapability::Client connectLocal2() {
	Temporary<RootConfig> config;
	return connectLocal1(config);
}

DynamicCapability::Client connectRemote1(kj::StringPtr address, unsigned int port) {
	LibraryThread lt = fscpy::PyContext::libraryThread();
	return connectRemote(lt, address, port);
}

}

namespace fscpy {

void loadDefaultSchema(py::module_& m) {
	defaultLoader.addBuiltin<ToroidalGrid, MagneticField, RootService>();
	auto schemas = getBuiltinSchemas<RootService>();
	
	/*for(auto node : schemas) {
		defaultLoader.add(node);
	}*/
	
	for(auto node : schemas) {
		defaultLoader.importNodeIfRoot(node.getId(), m);
	}
	
	m.def("connectLocal", &connectLocal1);
	m.def("connectLocal", &connectLocal2);
	m.def("connectLocal", &connectRemote1, py::arg("address"), py::arg("port") = 0);
}

}