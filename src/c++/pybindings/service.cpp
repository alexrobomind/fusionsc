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
#include <fsc/hint.capnp.h>
#include <fsc/dynamic.capnp.h>

#include "fsc-servepy.h"

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

LocalResources::Client connectSameThread1(LocalConfig::Reader config) {
	fscpy::PythonContext::start();
	return createLocalResources(config);
}
	
LocalResources::Client connectSameThread2() {
	Temporary<LocalConfig> config;
	return connectSameThread1(config);
}

/*struct LocalRootServer {
	Own<const InProcessServer> backend;
	
	LocalRootServer() :
		backend(
			newInProcessServer([]() {
				Temporary<LocalConfig> rootConfig;
				return createLocalResources(rootConfig);
			})
		)
	{}
	
	LocalResources::Client connect() {
		return backend -> connect<LocalResources>();
	}
};*/

py::object connectLocal() {
	return fsc::pybindings::createLocalServer([]{
		Temporary<LocalConfig> rootConfig;
		return createLocalResources(rootConfig);
	});
}

}

namespace fscpy {
	void initService(py::module_& m) {	
		m.def("connectSameThread", &connectSameThread1);
		m.def("connectSameThread", &connectSameThread2);
		m.def("connectLocal", &connectLocal);
		
		/*py::class_<LocalRootServer>(m, "LocalRootServer")
			.def(py::init<>())
			.def("connect", &LocalRootServer::connect)
		;*/
	}
}
