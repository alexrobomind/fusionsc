#include "fscpy.h"
#include "async.h"
#include "loader.h"

#include <fsc/data.h>

using namespace fscpy;

namespace {

PyPromise download(capnp::DynamicCapability::Client capability) {
	using capnp::AnyPointer;
	using capnp::DynamicCapability;
	using capnp::DynamicStruct;
	using capnp::DynamicList;
	using capnp::DynamicValue;
	
	fscpy::PyContext::startEventLoop();
	
	constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
	
	capnp::InterfaceSchema refSchema = capability.getSchema();
	
	KJ_REQUIRE(
		refSchema.getProto().getId() == DR_ID,
		"Can only download DataRef types"
	);
	
	DataRef<AnyPointer>::Client dataRef = capability.castAs<DataRef<AnyPointer>>();
	
	capnp::Type payloadType = refSchema.getBrandArgumentsAtScope(DR_ID)[0];
	KJ_REQUIRE(payloadType.isInterface() || payloadType.isStruct() || payloadType.isData(), "DataRefs can only carry interface, struct or data types");
	
	auto promise = getActiveThread().dataService().download(dataRef)
	.then([payloadType](LocalDataRef<AnyPointer> localRef) {
		// Because async execution usually happens outside the GIL, we need to re-acquire it here
		py::gil_scoped_acquire withPythonGIL;
		
		// Create a keepAlive object
		py::object keepAlive = py::cast((UnknownObject*) new UnknownHolder<LocalDataRef<AnyPointer>>(localRef));
		
		// "Data" type payloads do not have a root pointer. They are NO capnp messages.
		if(payloadType.isData()) {
			KJ_REQUIRE(localRef.getTypeID() == 0);
			
			py::object result = py::cast(localRef.getRaw());
			result.attr("_ref") = keepAlive;
			return kj::refcounted<PyObjectHolder>(mv(result));
		} else {
			AnyPointer::Reader root = localRef.get();
			
			if(payloadType.isInterface()) {
				auto schema = payloadType.asInterface();
				
				KJ_REQUIRE(localRef.getTypeID() == schema.getProto().getId());
				DynamicValue::Reader asDynamic = root.getAs<DynamicCapability>(schema);
				
				return kj::refcounted<PyObjectHolder>(py::cast(asDynamic));
			}
			if(payloadType.isStruct()) {
				auto schema = payloadType.asStruct();
				
				KJ_REQUIRE(localRef.getTypeID() == schema.getProto().getId());
				DynamicValue::Reader asDynamic = root.getAs<DynamicStruct>(schema);
				
				py::object result = py::cast(asDynamic);
				result.attr("_ref") = keepAlive;
				return kj::refcounted<PyObjectHolder>(mv(result));
			}
		}
		
		KJ_FAIL_REQUIRE("Internal error");
	}).eagerlyEvaluate(nullptr);
	
	return promise;
}

capnp::DynamicCapability::Client publish(capnp::DynamicStruct::Reader value) {
	auto schema = value.getSchema();
	uint64_t id = schema.getProto().getId();
	
	// Retrieve brand of published value
	Temporary<capnp::schema::Brand> valueBrand;
	extractBrand(schema, valueBrand);
	
	// Create brand for DataRef with single struct binding 
	constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
	Temporary<capnp::schema::Brand> brand;
	auto scope = brand.initScopes(1)[0];
	scope.setScopeId(DR_ID);
	auto boundStruct = scope.initBind(1)[0].initType().initStruct();
	
	// Bind single binding slot to published type
	boundStruct.setTypeId(id);
	boundStruct.setBrand(valueBrand.asReader());
	
	// Create branded DataRef schema
	auto dataRefSchema = defaultLoader.capnpLoader.get(DR_ID, brand.asReader());
	
	// Publish DataRef and convert to correct type
	capnp::Capability::Client asAny = getActiveThread().dataService().publish(value);
	return asAny.castAs<capnp::DynamicCapability>(dataRefSchema.asInterface());
}

capnp::DynamicValue::Reader openArchive(kj::StringPtr path) {
	using capnp::AnyPointer;
	using capnp::DynamicCapability;
	using capnp::DynamicStruct;
	using capnp::DynamicList;
	using capnp::DynamicValue;
	
	fscpy::PyContext::startEventLoop();
	
	//TODO: Put an injection context into Library
	auto fs = kj::newDiskFilesystem();
	
	auto absPath = fs->getCurrentPath().evalNative(path);
	auto file = fs->getRoot().openFile(absPath);
	
	LocalDataRef<AnyPointer> root = getActiveThread().dataService().publishArchive<AnyPointer>(*file);
	
	auto maybeSchema = defaultLoader.capnpLoader.tryGet(root.getTypeID());
	KJ_IF_MAYBE(pSchema, maybeSchema) {
		KJ_REQUIRE(pSchema->getProto().isStruct() || pSchema->getProto().isInterface(), "Can only load archives with struct or interface types");
		
		if(pSchema->getProto().getIsGeneric()) {
			KJ_LOG(WARNING, "Loading archive with generic type, assuming default brand", root.getTypeID());
		}
		
		capnp::Capability::Client asAny = root;
		
		// Create brand binding for DataRef
		constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
		Temporary<capnp::schema::Brand> brand;
		auto scopes = brand.initScopes(1);
		auto scope = scopes[0];
		scope.setScopeId(DR_ID);
		
		auto bindings = scope.initBind(1);
		auto boundType = bindings[0].initType();
		
		if(pSchema->getProto().isInterface()) {
			boundType.initInterface().setTypeId(root.getTypeID());
		} else {
			boundType.initStruct().setTypeId(root.getTypeID());
		}
		
		capnp::InterfaceSchema dataRefSchema = defaultLoader.capnpLoader.get(DR_ID, brand.asReader()).asInterface();
		
		return asAny.castAs<capnp::DynamicCapability>(dataRefSchema);
	}
	
	KJ_FAIL_REQUIRE("Failed to load schema for payload type ID", root.getTypeID());
}

Promise<void> writeArchive1(capnp::DynamicCapability::Client ref, kj::StringPtr path) {
	constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
	KJ_REQUIRE(ref.getSchema().getProto().getId() == DR_ID, "Can only publish capabilities of type DataRef");
	capnp::Capability::Client asAny = ref;
	auto asRef = asAny.castAs<DataRef<capnp::AnyPointer>>();
	
	auto fs = kj::newDiskFilesystem();
	
	auto absPath = fs->getCurrentPath().evalNative(path);
	auto file = fs->getRoot().openFile(absPath, kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
	
	return getActiveThread().dataService().writeArchive(asRef, *file).attach(mv(file));
}

Promise<void> writeArchive2(capnp::DynamicStruct::Reader root, kj::StringPtr path) {
	/*auto ref = getActiveThread().dataService().publish(getActiveThread().randomID(), root);
	
	auto fs = kj::newDiskFilesystem();
	
	auto absPath = fs->getCurrentPath().evalNative(path);
	auto file = fs->getRoot().openFile(absPath, kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
	
	return getActiveThread().dataService().writeArchive(ref, *file).attach(mv(file));*/
	return writeArchive1(publish(root), path);
}

Promise<void> writeArchive3(capnp::DynamicStruct::Builder root, kj::StringPtr path) {
	return writeArchive2(root.asReader(), path);
}

}

namespace fscpy {	
	
void initData(py::module_& m) {
	py::module_ dataModule = m.def_submodule("data", "Distributed data processing");
	
	dataModule.def("downloadAsync", &download, "Starts a download for the given DataRef and returns a promise for its contents");
	dataModule.def("publish", &publish, "Creates a DataRef containing the given data");
	
	dataModule.def("openArchive", &openArchive, "Opens an archive file and returns a DataRef to its root");
	
	dataModule.def("writeArchiveAsync", &writeArchive1, "Downloads (recursively) the given DataRef and asynchronously waits an Archive containing its contents. Must wait on the returned promise.");
	dataModule.def("writeArchiveAsync", &writeArchive2);
	dataModule.def("writeArchiveAsync", &writeArchive3);
}

}