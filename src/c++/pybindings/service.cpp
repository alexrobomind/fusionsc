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

struct LocalRootServer {
	kj::Function<capnp::Capability::Client()> clientFactory;
	
	LocalRootServer() :
		clientFactory(
			newInProcessServer<LocalResources>([]() mutable {
				Temporary<LocalConfig> rootConfig;
				return createLocalResources(rootConfig);
			})
		)
	{}
	
	LocalResources::Client connect() {
		auto result = clientFactory().castAs<LocalResources>();
		return result;
	}
};

}

namespace fscpy {
	void initService(py::module_& m) {	
		m.def("connectSameThread", &connectSameThread1);
		m.def("connectSameThread", &connectSameThread2);
		
		py::class_<LocalRootServer>(m, "LocalRootServer")
			.def(py::init<>())
			.def("connect", &LocalRootServer::connect)
		;
	}

	void loadDefaultSchema(py::module_& m) {
		/*
		// Here we need to specify datatypes that need to be loaded because they are passed to the python interface
		
		#define FSC_BUILTIN_SCHEMAS FieldResolver, GeometryResolver, NetworkInterface, LocalResources, OfflineData, \
			MergedGeometry, FLTStopReason, HintEquilibrium, ReversibleFieldlineMapping, \
			SimpleHttpServer, Warehouse, Warehouse::Folder, DynamicObject
		
		defaultLoader.addBuiltin<
			capnp::schema::Node,
			FSC_BUILTIN_SCHEMAS
		>();
		
		// Schema submodule
		{
			auto schemas = getBuiltinSchemas<capnp::schema::Node>();
			py::module_ subM = m.def_submodule("schema", "Cap'n'proto structs describing Cap'n'proto types and schemas");
			
			for(auto node : schemas) {
				defaultLoader.importNodeIfRoot(node.getId(), subM);
			}
		}
	
		py::dict defaultGlobalScope;
		
		auto ires = py::module::import("importlib_resources");
		auto fscRoot = ires.attr("files")("fusionsc").attr("joinpath")("service");
		
		defaultGlobalScope["fusionsc"] = fscRoot.attr("joinpath")("fusionsc");
		defaultGlobalScope["capnp"] = fscRoot.attr("joinpath")("capnp");
		
		py::module_ schemaMod = m.def_submodule("schema", "Cap'n'proto structs describing Cap'n'proto types and schemas");
		py::module_ serviceMod = m.def_submodule("service", "Cap'n'proto service interface for fusionsc library");
		
		auto filesDir = fscRoot.attr("joinpath")("service/capnp");
		for(py::object name : filesDir.attr("iterdir")()) {
			parseSchema(filesDir, name, schemaMod, defaultGlobalScope);
		}
		
		filesDir = fscRoot.attr("joinpath")("service/fusionsc");
		for(py::object name : filesDir.attr("iterdir")()) {
			parseSchema(filesDir, name, serviceMod, defaultGlobalScope);
		}*/
	}
}