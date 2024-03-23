#include "fscpy.h"

using capnp::StructSchema;

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

py::type Loader::commonType(capnp::Type type) {
	using Which = capnp::schema::Type::Which;
	
	switch(type.which()) {
		case Which::TEXT: return py::type::of<TextCommon>();
		case Which::DATA: return py::type::of<DataCommon>();
		case Which::LIST: return py::type::of<DynamicListCommon>();
		case Which::ANY_POINTER: return py::type::of<AnyCommon>();
		
		case Which::STRUCT: break;
		
		default: KJ_FAIL_REQUIRE("Builders can only be requested for pointer types");
	}
	
	KJ_IF_MAYBE(pResult, commonTypes.find(type)) {
		return *pResult;
	}
	
	auto qn = qualName(type);
	
	py::dict attributes;
	attributes["__qualname__"] = kj::get<1>(qn).flatten().cStr();
	attributes["__module__"] = kj::get<0>(qn).cStr();
	
	py::type cls = (*baseMetaType)("ReaderOrBuilder", py::make_tuple(), attributes);
	return cls;
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

py::type makeType(Loader& loader, capnp::Schema schema, kj::StringPtr kind, py::type primaryBase, py::tuple bases, py::dict attributes) {
	auto qn = loader.qualName(schema);
	kj::StringPtr moduleName = kj::get<0>(qn);
	kj::String className = kj::get<1>(qn).flatten();
	
	kj::StringTree docString = kj::strTree(kind, " for ", className);
	
	// Add docString from schema source
	auto maybeSource = loader.sourceInfo.find(schema.getProto().getId());
	KJ_IF_MAYBE(pSrc, maybeSource) {
		auto comment = pSrc -> getDocComment();
		if(comment.size() > 0) {
			docString = kj::strTree(
				mv(docString),
				"\n\n --- Documentation for ", className, " ---\n\n",
				comment
			);
		}
	}
	
	// Add constructor delegating to primary base		
	attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
		[primaryBase](py::object self, py::args args, py::kwargs kwargs) {
			primaryBase.attr("__init__")(self, *args, **kwargs);
		}
	));
	
	// Set qualname and module;
	attributes["__module__"] = moduleName.cStr();
	attributes["__qualname__"] = className.cStr();
	
	// Set docstring
	attributes["__doc__"] = py::str(docString.flatten().cStr());
	
	// Disable additional attributes
	attributes["__slots__"] = py::make_tuple();
	
	py::type newCls = (*baseMetaType)(kind.cStr(), bases, attributes);
	return newCls;
}

py::type Loader::makeReaderType(capnp::StructSchema schema) {
	auto maybeSource = sourceInfo.find(schema.getProto().getId());
	py::dict attributes;
		
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr rawName = field.getProto().getName();
		kj::String name = memberName(rawName);
		
		using Field = capnp::schema::Field;
		
		auto type = field.getType();
				
		kj::String docString;
		KJ_IF_MAYBE(pSrc, maybeSource) {
			auto comment = pSrc -> getMembers()[field.getIndex()].getDocComment().asReader();
			if(comment.size() > 0) {
				docString = kj::heapString(comment);
			}
		}
		
		attributes[name.cStr()] = FieldDescriptor(field, mv(docString));
	}
	
	py::type baseClass = py::type::of<DynamicStructReader>();
	
	return makeType(
		*this,
		schema, "Reader",
		baseClass, py::make_tuple(baseClass, this -> commonType(schema)),
		attributes
	);
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
