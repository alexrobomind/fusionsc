#include "fscpy.h"

namespace fscpy {

py::type Loader::builderType(capnp::Type type) {
	KJ_IF_MAYBE(pResult, builderTypes.find(type)) {
		return *pResult;
	}
	auto result = makeBuilderType(type);
	builderTypes.insert(type, result);
	return result;
}

py::type Loader::readerType(capnp::Type type) {
	KJ_IF_MAYBE(pResult, readerTypes.find(type)) {
		return *pResult;
	}
	auto result = makeReaderType(type);
	readerTypes.insert(type, result);
	return result;
}

py::type Loader::pipelineType(capnp::StructSchema type) {
	KJ_IF_MAYBE(pResult, pipelineTypes.find(type)) {
		return *pResult;
	}
	auto result = makePipelineType(type);
	pipelineTypes.insert(type, result);
	return result;
}

py::type Loader::clientType(capnp::InterfaceSchema type) {
	KJ_IF_MAYBE(pResult, clientTypes.find(type)) {
		return *pResult;
	}
	auto result = makeClientType(type);
	clientTypes.insert(type, result);
	return result;
}

py::type Loader::makeBuilderType(capnp::Type type) {
	using Which = capnp::schema::Type::Which;
	
	switch(type.which()) {
		case Which::TEXT: return py::type::of<TextBuilder>();
		case Which::DATA: return py::type::of<DataBuilder>();
		case Which::STRUCT: return makeBuilderType(type.asStruct());
		case Which::LIST: return py::type::of<DynamicListBuilder>();
		case Which::ANY_POINTER: return py::type::of<AnyBuilder>();
		
		default: KJ_FAIL_REQUIRE("Builders can only be requested for pointer types");
	}
	
	KJ_UNREACHABLE;
}

py::type Loader::makeReaderType(capnp::Type type) {
	using Which = capnp::schema::Type::Which;
	
	switch(type.which()) {
		case Which::TEXT: return py::type::of<TextReader>();
		case Which::DATA: return py::type::of<DataReader>();
		case Which::STRUCT: return makeReaderType(type.asStruct());
		case Which::LIST: return py::type::of<DynamicListReader>();
		case Which::ANY_POINTER: return py::type::of<AnyReader>();
		
		default: KJ_FAIL_REQUIRE("Builders can only be requested for pointer types");
	}
	
	KJ_FAIL_REQUIRE("Unknown type kind");
}

py::type Loader::makeReaderType(capnp::StructSchema) {
	KJ_UNIMPLEMENTED();
}


py::type Loader::makeBuilderType(capnp::StructSchema) {
	KJ_UNIMPLEMENTED();
}

py::type Loader::makePipelineType(capnp::StructSchema) {
	KJ_UNIMPLEMENTED();
}

py::type Loader::makeClientType(capnp::InterfaceSchema) {
	KJ_UNIMPLEMENTED();
}


}
