#include "fscpy.h"
#include "async.h"
#include "loader.h"
#include "data.h"

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
	
DynamicValueReader openRef(capnp::Type payloadType, LocalDataRef<> dataRef) {	
	// "Data" type payloads do not have a root pointer. They are NO capnp messages.
	if(dataRef.getFormat().isRaw()) {
		return DataReader::from(dataRef.forkRaw());
	}
	
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
	
	capnp::AnyPointer::Reader root = dataRef.get();
	
	if(payloadType.isInterface()) {
		auto schema = payloadType.asInterface();
		return root.getAs<capnp::DynamicCapability>(schema);
	}
	if(payloadType.isStruct()) {
		auto schema = payloadType.asStruct();
		return DynamicStructReader(kj::heap(dataRef), root.getAs<capnp::DynamicStruct>(schema));
	}
	if(payloadType.isData()) {
		return DataReader(kj::heap(dataRef), root.getAs<capnp::Data>());
	}
	if(payloadType.isList()) {
		auto schema = payloadType.asList();
		return DynamicListReader(kj::heap(dataRef), root.getAs<capnp::DynamicList>(schema));
	}
	
	KJ_FAIL_REQUIRE("DataRefs can only carry pointer types (interface, struct, data, text)");
}
	
Promise<DynamicValueReader> download(capnp::DynamicCapability::Client capability) {
	using capnp::AnyPointer;
	using capnp::DynamicCapability;
	using capnp::DynamicStruct;
	using capnp::DynamicList;
	using capnp::DynamicValue;
	
	DataRef<AnyPointer>::Client dataRef = capability.castAs<DataRef<AnyPointer>>();
	auto promise = getActiveThread().dataService().download(dataRef)
	.then([payloadType = getRefPayload(capability.getSchema())](LocalDataRef<AnyPointer> localRef) mutable {
		return openRef(payloadType, mv(localRef));
	}).eagerlyEvaluate(nullptr);
	
	return promise;
}

DynamicCapabilityClient publishReader(DynamicValueReader value) {
	capnp::Type baseType;
	
	auto& ds = getActiveThread().dataService();
	
	// Publish DataRef and convert to correct type
	capnp::Capability::Client asAny = nullptr;
	
	switch(value.getType()) {
		case capnp::DynamicValue::DATA:
			asAny = ds.publish(value.asData().wrapped());
			baseType = capnp::Type::from<capnp::Data>();
			break;
		case capnp::DynamicValue::STRUCT:
			asAny = ds.publish(value.asStruct().wrapped());
			baseType = value.asStruct().getSchema();
			break;
		default:
			KJ_FAIL_REQUIRE("Can only publish data blobs and struct readers");
	}
	
	auto dataRefSchema = createRefSchema(baseType);
	return asAny.castAs<capnp::DynamicCapability>(dataRefSchema);
}

DynamicCapabilityClient publishBuilder(DynamicValueBuilder dsb) {
	return publishReader(dsb);
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

Promise<void> writeArchive2(DynamicStructReader root, kj::StringPtr path) {
	return writeArchive1(publishReader(root), path);
}

Promise<void> writeArchive3(DynamicStructBuilder root, kj::StringPtr path) {
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