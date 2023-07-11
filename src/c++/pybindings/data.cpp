#include "fscpy.h"
#include "async.h"
#include "loader.h"

#include <fsc/data.h>
#include <fsc/services.h>

using namespace fscpy;

namespace fscpy {
	
capnp::Type getRefPayload(capnp::InterfaceSchema refSchema) {
	constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
	
	KJ_REQUIRE(
		refSchema.getProto().getId() == DR_ID,
		"Type must be a DataRef instance"
	);
	
	return refSchema.getBrandArgumentsAtScope(DR_ID)[0];
}

capnp::InterfaceSchema createRefSchema(capnp::Type payloadType) {
	constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
	Temporary<capnp::schema::Brand> brand;
	auto scope = brand.initScopes(1)[0];
	scope.setScopeId(DR_ID);
	
	extractType(payloadType, scope.initBind(1)[0].initType());
	
	return defaultLoader.capnpLoader.get(DR_ID, brand.asReader()).asInterface();
}

Maybe<capnp::Type> getPayloadType(LocalDataRef<> dataRef) {
	auto format = dataRef.getFormat();
	
	if(format.isRaw()) {
		return capnp::Type(capnp::schema::Type::DATA);
	} else if(format.isSchema()) {
		return defaultLoader.capnpLoader.getType(format.getSchema().getAs<capnp::schema::Type>());
	} else if(format.isUnknown()) {
		return nullptr;
	} else {
		KJ_FAIL_REQUIRE("Unknown payload type");
	}
}
	
kj::Own<PyObjectHolder> openRef(capnp::Type payloadType, LocalDataRef<> dataRef) {
	// Because async execution usually happens outside the GIL, we need to re-acquire it here
	py::gil_scoped_acquire withPythonGIL;
	
	// Create a keepAlive object
	py::object keepAlive = py::cast((UnknownObject*) new UnknownHolder<LocalDataRef<>>(dataRef));
	
	auto inferredType = getPayloadType(dataRef);
	if(payloadType.isAnyPointer()) {
		KJ_IF_MAYBE(pType, inferredType) {
			payloadType = *pType;
		} else {
			KJ_FAIL_REQUIRE("Could not deduce the content type. A static type was not given (or set as AnyPointer) and the format type of the loaded data is 'unknown'");
		}
	} else {
		KJ_IF_MAYBE(pType, inferredType) {
			KJ_REQUIRE(payloadType == *pType, "Static and dynamic type do not match");
		}
	}
	
	// "Data" type payloads do not have a root pointer. They are NO capnp messages.
	if(payloadType.isData()) {		
		py::object result = py::cast(dataRef.getRaw());
		result.attr("_ref") = keepAlive;
		return kj::refcounted<PyObjectHolder>(mv(result));
	} else {
		capnp::AnyPointer::Reader root = dataRef.get();
		
		if(payloadType.isInterface()) {
			auto schema = payloadType.asInterface();
			
			capnp::DynamicValue::Reader asDynamic = root.getAs<capnp::DynamicCapability>(schema);
			return kj::refcounted<PyObjectHolder>(py::cast(asDynamic));
		}
		if(payloadType.isStruct()) {
			auto schema = payloadType.asStruct();
			
			capnp::DynamicValue::Reader asDynamic = root.getAs<capnp::DynamicStruct>(schema);
			
			py::object result = py::cast(asDynamic);
			result.attr("_ref") = keepAlive;
			return kj::refcounted<PyObjectHolder>(mv(result));
		}
		
		KJ_FAIL_REQUIRE("DataRefs can only carry interface, struct or data types (or AnyPointer if unknown)");
	}
}
	
PyPromise download(capnp::DynamicCapability::Client capability) {
	using capnp::AnyPointer;
	using capnp::DynamicCapability;
	using capnp::DynamicStruct;
	using capnp::DynamicList;
	using capnp::DynamicValue;
	
	fscpy::PythonContext::startEventLoop();
	
	DataRef<AnyPointer>::Client dataRef = capability.castAs<DataRef<AnyPointer>>();
	auto promise = getActiveThread().dataService().download(dataRef)
	.then([payloadType = getRefPayload(capability.getSchema())](LocalDataRef<AnyPointer> localRef) mutable {
		return openRef(payloadType, mv(localRef));
	}).eagerlyEvaluate(nullptr);
	
	return promise;
}

capnp::DynamicCapability::Client publishReader(capnp::DynamicStruct::Reader value) {	
	auto dataRefSchema = createRefSchema(value.getSchema());
	
	// Publish DataRef and convert to correct type
	capnp::Capability::Client asAny = getActiveThread().dataService().publish(value);
	return asAny.castAs<capnp::DynamicCapability>(dataRefSchema);
}

capnp::DynamicCapability::Client publishBuilder(capnp::DynamicStruct::Builder dsb) {
	return publishReader(dsb.asReader());
}

}

namespace {
	
capnp::DynamicCapability::Client openArchive(kj::StringPtr path, LocalResources::Client localResources) {
	using capnp::AnyPointer;
	using capnp::DynamicCapability;
	using capnp::DynamicStruct;
	using capnp::DynamicList;
	using capnp::DynamicValue;
	
	// Open archive in this thread to determine type
	//TODO: Put an injection context into Library
	auto fs = kj::newDiskFilesystem();
	auto absPath = fs->getCurrentPath().evalNative(path);
	auto file = fs->getRoot().openFile(absPath);
	
	LocalDataRef<AnyPointer> root = getActiveThread().dataService().publishArchive<AnyPointer>(*file);
	
	capnp::Type type = capnp::Type(capnp::schema::Type::DATA);
	auto format = root.getFormat();
	if(format.isSchema()) {
		type = defaultLoader.capnpLoader.getType(format.getSchema().getAs<capnp::schema::Type>());
	} else if(format.isUnknown()) {
		KJ_FAIL_REQUIRE("Internal error: The root of the archive file has no format specified. This should not be the case");
	}
	
	auto dataRefSchema = createRefSchema(type);
	
	// Use localResources to open ref in other thread
	auto request = localResources.openArchiveRequest();
	request.setFilename(path);
	capnp::Capability::Client asAny = request.sendForPipeline().getRef();
	
	return asAny.castAs<capnp::DynamicCapability>(dataRefSchema);
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
	return writeArchive1(publishReader(root), path);
}

Promise<void> writeArchive3(capnp::DynamicStruct::Builder root, kj::StringPtr path) {
	return writeArchive1(publishBuilder(root), path);
}

}

namespace fscpy {	
	
void initData(py::module_& m) {
	py::module_ dataModule = m.def_submodule("data", "Distributed data processing");
	
	dataModule.def("downloadAsync", &download, "Starts a download for the given DataRef and returns a promise for its contents");
	dataModule.def("publish", &publishBuilder, "Creates a DataRef containing the given data");
	dataModule.def("publish", &publishReader, "Creates a DataRef containing the given data");
	
	dataModule.def("openArchive", &openArchive, "Opens an archive file and returns a DataRef to its root");
	
	dataModule.def("writeArchiveAsync", &writeArchive1, "Downloads (recursively) the given DataRef and asynchronously waits an Archive containing its contents. Must wait on the returned promise.");
	dataModule.def("writeArchiveAsync", &writeArchive2);
	dataModule.def("writeArchiveAsync", &writeArchive3);
}

}