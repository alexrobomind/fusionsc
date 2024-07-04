#include "pickle.h"
#include "data.h"
#include "async.h"
#include "loader.h"

#include <capnp/serialize-packed.h>

namespace fscpy {
	
namespace {

kj::Array<const byte> fromPythonBuffer(py::buffer buf) {
	auto bufInfo = buf.request();
	
	kj::ArrayPtr<const byte> ptr((const byte*) bufInfo.ptr, bufInfo.itemsize * bufInfo.size);
	
	// Add a deleter for the buffer info that deletes inside the GIL
	Maybe<py::buffer_info> deletable = mv(bufInfo);
	return ptr.attach(kj::defer([deletable = mv(deletable)]() mutable {
		py::gil_scoped_acquire withGil;
		deletable = nullptr;
	}));
}

py::list flattenDataRef(uint32_t pickleVersion, capnp::DynamicCapability::Client dynamicRef) {	
	DataRef<>::Client staticRef = dynamicRef.castAs<DataRef<>>();
	kj::Array<kj::Array<const byte>> data = PythonWaitScope::wait(getActiveThread().dataService().downloadFlat(staticRef));
	
	py::list result(data.size());
	
	if(pickleVersion <= 4) {
		// Version with copying
		for(auto i : kj::indices(data)) {
			result[i] = py::bytes((const char*) data[i].begin(), (uint64_t) data[i].size());
		}
	} else {
		// Zero-copy version
		auto pbCls = py::module_::import("pickle").attr("PickleBuffer");
		for(auto i : kj::indices(data)) {			
			result[i] = pbCls(DataReader::from(mv(data[i])));
		}
	}
	
	return result;
}

LocalDataRef<> unflattenDataRef(py::list input) {
	auto arrayBuilder = kj::heapArrayBuilder<kj::Array<const byte>>(input.size());
	
	for(auto i : kj::indices(input)) {
		arrayBuilder.add(fromPythonBuffer(py::reinterpret_borrow<py::buffer>(input[i])));
	}
	
	return getActiveThread().dataService().publishFlat<capnp::AnyPointer>(arrayBuilder.finish());
}

}

py::object pickleReduceReader(DynamicStructReader reader, uint32_t pickleVersion) {
	auto unpickler = py::module_::import("fusionsc").attr("pickle_support").attr("unpickleReader");
	return py::make_tuple(
		unpickler,
		py::make_tuple(
			1,
			flattenDataRef(pickleVersion, publishReader(reader))
		)
	);
}
py::object pickleReduceBuilder(DynamicStructBuilder builder, uint32_t pickleVersion) {
	auto unpickler = py::module_::import("fusionsc").attr("pickle_support").attr("unpickleBuilder");
	return py::make_tuple(
		unpickler,
		py::make_tuple(
			1,
			flattenDataRef(pickleVersion, publishBuilder(builder))
		)
	);
}

py::object pickleReduceRef(capnp::DynamicCapability::Client clt, uint32_t pickleVersion) {
	auto unpickler = py::module_::import("fusionsc").attr("pickle_support").attr("unpickleRef");
	return py::make_tuple(
		unpickler,
		py::make_tuple(
			1,
			flattenDataRef(pickleVersion, mv(clt))
		)
	);
}

py::object pickleReduceEnum(EnumInterface enumerant, uint32_t pickleVersion) {
	auto schema = enumerant.getSchema();
	
	// Store type information
	py::object typeInfo;
	if(schema.getGenericScopeIds().size() == 0) {
		// For unbranded enums, we just store the node ID
		// This should be extremely common
		typeInfo = py::int_(schema.getProto().getId());
	} else {
		KJ_UNIMPLEMENTED(
			"Currently the pickling path for generic enums is untested. Please contact "
			"a.knieps@fz-juelich.de with your use case so that we can use it to test "
			"this code path before enabling it."
		);

		// For branded enums, we need to store the entire
		// type information struct. We write it packed to save
		// space.
		Temporary<capnp::schema::Type> typeInfoRaw;
		auto asEnum = typeInfoRaw.initEnum();
		asEnum.setTypeId(enumerant.getSchema().getProto().getId());
		extractBrand(enumerant.getSchema(), asEnum.initBrand());
		
		// Serialize to flat array
		kj::VectorOutputStream os;
		capnp::writePackedMessage(os, *typeInfoRaw.holder);
		
		auto outputBuffer = os.getArray();
		typeInfo = py::bytes((const char*) outputBuffer.begin(), (uint64_t) outputBuffer.size());
	}
	
	auto unpickler = py::module_::import("fusionsc").attr("pickle_support").attr("unpickleEnum");
	return py::make_tuple(
		unpickler,
		py::make_tuple(
			1,
			typeInfo,
			enumerant.getRaw()
		)
	);
}

DynamicValueBuilder unpickleBuilder(uint32_t version, py::list data) {
	return unpickleReader(version, mv(data)).clone();
}

DynamicValueReader unpickleReader(uint32_t version, py::list data) {	
	KJ_REQUIRE(version == 1, "Only version 1 representation supported");
	auto ref = unflattenDataRef(data);
	return openRef(capnp::schema::Type::AnyPointer::Unconstrained::STRUCT, mv(ref));
}

DynamicCapabilityClient unpickleRef(uint32_t version, py::list data) {
	KJ_REQUIRE(version == 1, "Only version 1 representation supported");
	return unflattenDataRef(data);
}

EnumInterface unpickleEnum(uint32_t version, py::object typeInfo, uint16_t value) {
	KJ_REQUIRE(version == 1, "Only version 1 representation supported");
	
	capnp::EnumSchema schemaBase;
	if(py::isinstance<py::int_>(typeInfo)) {
		uint64_t typeId = py::int_(mv(typeInfo));
		schemaBase = defaultLoader.capnpLoader.get(typeId).asEnum();
	} else {
		py::bytes inputBytes(mv(typeInfo));
		std::string data = inputBytes;
		
		kj::ArrayInputStream inputStream(kj::ArrayPtr<const byte>((const unsigned char*) data.data(), data.size()));
		capnp::PackedMessageReader msg(inputStream);
		
		auto root = msg.getRoot<capnp::schema::Type>();
		auto asEnum = root.getEnum();
		
		schemaBase = defaultLoader.capnpLoader.get(asEnum.getTypeId(), asEnum.getBrand()).asEnum();
	}
	
	return capnp::DynamicEnum(mv(schemaBase), value);
}

}