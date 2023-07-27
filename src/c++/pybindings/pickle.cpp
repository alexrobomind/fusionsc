#include "pickle.h"
#include "data.h"
#include "async.h"

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
	auto payloadType = getRefPayload(dynamicRef.getSchema());
	
	auto data = PythonWaitScope::wait(getActiveThread().dataService().downloadFlat(dynamicRef.castAs<DataRef<>>()));
	
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

}