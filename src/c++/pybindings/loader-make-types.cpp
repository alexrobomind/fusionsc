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

py::type Loader::serverType(capnp::InterfaceSchema type) {
	KJ_IF_MAYBE(pResult, serverTypes.find(type)) {
		return *pResult;
	}
	auto result = makeServerType(type);
	serverTypes.insert(type, result);
	return result;
}

py::type Loader::commonType(capnp::Type type) {
	KJ_IF_MAYBE(pResult, commonTypes.find(type)) {
		return *pResult;
	}
	auto result = makeCommonType(type);
	commonTypes.insert(type, result);
	return result;
}

py::type Loader::makeCommonType(capnp::Type type) {
	using Which = capnp::schema::Type::Which;
	
	switch(type.which()) {
		case Which::TEXT: return py::type::of<TextCommon>();
		case Which::DATA: return py::type::of<DataCommon>();
		case Which::STRUCT: break;
		case Which::LIST: return py::type::of<DynamicListCommon>();
		case Which::ANY_POINTER: {
			using Which = capnp::schema::Type::AnyPointer::Unconstrained::Which;
			switch(type.whichAnyPointerKind()) {
				case Which::ANY_KIND: return py::type::of<AnyCommon>();
				case Which::CAPABILITY: return py::type::of<DynamicCapabilityClient>();
				case Which::STRUCT: return py::type::of<DynamicStructCommon>();
				case Which::LIST: return py::type::of<DynamicListCommon>();
			}
		}
		
		case Which::VOID: return py::type::of(py::none());
		case Which::BOOL: return py::type::of(py::bool_(true));
		
		case Which::UINT8:
		case Which::UINT16:
		case Which::UINT32:
		case Which::UINT64:
		case Which::INT8:
		case Which::INT16:
		case Which::INT32:
		case Which::INT64:
			return py::type::of(py::int_(1));
		
		case Which::FLOAT32:
		case Which::FLOAT64:
			return py::type::of(py::float_(0.0));
			
		
		case Which::INTERFACE:
			return clientType(type.asInterface());
	}
	
	KJ_IF_MAYBE(pResult, commonTypes.find(type)) {
		return *pResult;
	}
	
	auto qn = qualName(type);
	
	py::dict attributes;
	py::tuple bases = py::make_tuple();
	if(type.asStruct().isBranded()) {
		bases = py::make_tuple(commonType(type.asStruct().getGeneric().asStruct()));
	}
	
	attributes["__qualname__"] = kj::str(kj::get<1>(qn).flatten(), ".ReaderOrBuilder").cStr();
	attributes["__module__"] = kj::get<0>(qn).cStr();
	
	py::type cls = (*baseMetaType)("ReaderOrBuilder", bases, attributes);
	return cls;
}

py::type Loader::makeBuilderType(capnp::Type type) {
	using Which = capnp::schema::Type::Which;
	
	switch(type.which()) {
		case Which::TEXT: return py::type::of<TextBuilder>();
		case Which::DATA: return py::type::of<DataBuilder>();
		case Which::STRUCT: return makeBuilderType(type.asStruct());
		case Which::LIST: return makeBuilderType(type.asList());
		case Which::ANY_POINTER: 
			using Which = capnp::schema::Type::AnyPointer::Unconstrained::Which;
			switch(type.whichAnyPointerKind()) {
				case Which::ANY_KIND: return py::type::of<AnyBuilder>();
				case Which::CAPABILITY: return py::type::of<DynamicCapabilityClient>();
				case Which::STRUCT: return py::type::of<DynamicStructBuilder>();
				case Which::LIST: return py::type::of<DynamicListBuilder>();
			}
		
		default: return commonType(type);
	}
	
	KJ_UNREACHABLE;
}

py::type Loader::makeReaderType(capnp::Type type) {
	using Which = capnp::schema::Type::Which;
	
	switch(type.which()) {
		case Which::TEXT: return py::type::of<TextReader>();
		case Which::DATA: return py::type::of<DataReader>();
		case Which::STRUCT: return makeReaderType(type.asStruct());
		case Which::LIST: return makeReaderType(type.asList());
		case Which::ANY_POINTER: 
			using Which = capnp::schema::Type::AnyPointer::Unconstrained::Which;
			switch(type.whichAnyPointerKind()) {
				case Which::ANY_KIND: return py::type::of<AnyReader>();
				case Which::CAPABILITY: return py::type::of<DynamicCapabilityClient>();
				case Which::STRUCT: return py::type::of<DynamicStructReader>();
				case Which::LIST: return py::type::of<DynamicListReader>();
			}
		
		default: return commonType(type);
	}
	
	KJ_FAIL_REQUIRE("Unknown type kind");
}

namespace {

	py::type makeType(Loader& loader, capnp::Schema schema, kj::StringPtr kind, py::tuple bases, py::dict attributes) {
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
			[primaryBase = bases[0]](py::object self, py::args args, py::kwargs kwargs) {
				primaryBase.attr("__init__")(self, *args, **kwargs);
			}
		));
		
		// Set qualname and module;
		attributes["__module__"] = moduleName.cStr();
		attributes["__qualname__"] = kj::str(className, ".", kind).cStr();
		
		// Set docstring
		attributes["__doc__"] = py::str(docString.flatten().cStr());
		
		// Disable additional attributes
		attributes["__slots__"] = py::make_tuple();
		
		py::type newCls = (*baseMetaType)(kind.cStr(), bases, attributes);
		return newCls;
	}

	py::type makeListType(Loader& loader, capnp::ListSchema schema, py::type base, py::dict attributes, kj::StringPtr kind) {
		auto qn = loader.qualName(schema);
		kj::StringPtr moduleName = kj::get<0>(qn);
		kj::String className = kj::get<1>(qn).flatten();
		
		// Add constructor delegating to primary base		
		attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
			[base](py::object self, py::args args, py::kwargs kwargs) {
				base.attr("__init__")(self, *args, **kwargs);
			}
		));
		
		// Set qualname and module;
		attributes["__module__"] = moduleName.cStr();
		attributes["__qualname__"] = kj::str(className, ".", kind).cStr();
		
		// Disable additional attributes
		attributes["__slots__"] = py::make_tuple();
		
		py::type newCls = (*baseMetaType)(kind.cStr(), py::make_tuple(base), attributes);
		return newCls;
	}

	py::object fieldDescriptor(Loader& loader, capnp::StructSchema::Field field, kj::StringPtr suffix, kj::StringPtr docString) {
		py::options opts;
		opts.disable_function_signatures();
		
		// Note: This is a bit funky overall, because py::cpp_function has the "method descriptor" functionality
		// built in (apparently?), but not the associated data descriptor implementation. This means that the
		// __get__ function has to accept the self pointer to the descriptor object, but the other two don't.
		
		py::dict attributes;
		attributes["__get__"] = py::cpp_function(
			[field, doc = kj::heapString(docString)](py::object descriptorSelf, py::object objSelf, py::object objType) -> py::object {
				if(objSelf.is_none())
					return py::cast(doc.cStr());
				
				return objSelf[field.getProto().getName().cStr()];
			},
			kj::str("__get__(objSelf: Any, type: Any) -> ", loader.fullName(field.getType()), ".", suffix).cStr(),
			py::arg("descSelf"), py::arg("objSelf"), py::arg("type") = py::none()
		);
		
		attributes["__set__"] = py::cpp_function(
			[field](py::object objSelf, py::object value) {
				objSelf[field.getProto().getName().cStr()] = value;
			},
			kj::str("__set__(objSelf: Any, value: Any) -> None").cStr(),
			py::arg("objSelf"), py::arg("value")
		);
		
		attributes["__delete__"] = py::cpp_function(
			[field](py::object objSelf) {
				objSelf.attr("__delitem__")(field.getProto().getName().cStr());
			},
			kj::str("__delete__(objSelf: Any) -> None").cStr(),
			py::arg("objSelf")
		);
		
		attributes["__doc__"] = docString.cStr();
		
		py::type descriptorClass = py::eval("type")("TypedFieldDescriptor", py::make_tuple(), attributes);
		return descriptorClass();
	}
}

py::type Loader::makeReaderType(capnp::ListSchema schema) {	
	auto elementType = this -> readerType(schema.getElementType());
	auto elModuleName = py::cast<kj::StringPtr>(elementType.attr("__module__"));
	auto elClassName = py::cast<kj::StringPtr>(elementType.attr("__qualname__"));
	
	py::dict attributes;
	
	// Make new method for getitem and init that return proper types
	// that is properly typed
	{
		py::options opts;
		opts.disable_function_signatures();
		
		attributes["__getitem__"] = fscpy::methodDescriptor(py::cpp_function(
			[](DynamicListReader& self, uint32_t i) { return self[i]; },
			kj::str(
				"__getitem__: (self: ", fullName(schema), ".Reader, idx: int)",
				" -> ",
				elModuleName, ".", elClassName
			).cStr()
		));
	}
	
	return makeListType(*this, schema, py::type::of<DynamicListReader>(), attributes, "Reader");
}

py::type Loader::makeBuilderType(capnp::ListSchema schema) {	
	auto elementType = this -> builderType(schema.getElementType());
	auto elModuleName = py::cast<kj::StringPtr>(elementType.attr("__module__"));
	auto elClassName = py::cast<kj::StringPtr>(elementType.attr("__qualname__"));
	
	py::dict attributes;
	
	// Make new method for getitem and init that return proper types
	// that is properly typed
	{
		py::options opts;
		opts.disable_function_signatures();
		
		attributes["__getitem__"] = fscpy::methodDescriptor(py::cpp_function(
			[](DynamicListReader& self, uint32_t i) { return self[i]; },
			kj::str(
				"__getitem__: (self: ", fullName(schema), ".Reader, idx: int)",
				" -> ",
				elModuleName, ".", elClassName
			).cStr()
		));
		
		attributes["init"] = fscpy::methodDescriptor(py::cpp_function(
			[](DynamicListReader& self, uint32_t i) { return self[i]; },
			kj::str(
				"init: (self: ", fullName(schema), ".Reader, idx: int)",
				" -> ",
				elModuleName, ".", elClassName
			).cStr()
		));
	}
	
	return makeListType(*this, schema, py::type::of<DynamicListBuilder>(), attributes, "Builder");
}

py::type Loader::makeReaderType(capnp::StructSchema schema) {
	auto maybeSource = sourceInfo.find(schema.getProto().getId());
	py::dict attributes;
		
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr rawName = field.getProto().getName();
		kj::String name = memberName(rawName);
		
		using Field = capnp::schema::Field;
		
		auto type = field.getType();
				
		kj::String docString = kj::str("");
		KJ_IF_MAYBE(pSrc, maybeSource) {
			auto comment = pSrc -> getMembers()[field.getIndex()].getDocComment().asReader();
			if(comment.size() > 0) {
				docString = kj::heapString(comment);
			}
		}
		
		attributes[name.cStr()] = fieldDescriptor(*this, field, "Reader", docString);
	}
	
	py::type baseClass = schema.isBranded() ? readerType(schema.getGeneric().asStruct()) : py::type::of<DynamicStructReader>();
	
	return makeType(
		*this,
		schema, "Reader",
		py::make_tuple(baseClass, this -> commonType(schema)),
		attributes
	);
}

py::type Loader::makeBuilderType(capnp::StructSchema schema) {
	auto maybeSource = sourceInfo.find(schema.getProto().getId());
	
	py::dict attributes;
	
	auto addInitializer = [&](capnp::StructSchema::Field field) {
		auto type = field.getType();
		
		kj::StringPtr rawName = field.getProto().getName();
		kj::String name = memberName(rawName);
		
		kj::String nameUpper = kj::heapString(rawName);
		nameUpper[0] = toupper(rawName[0]);
		
		auto initializerName = str("init", nameUpper);
		
		py::options opts;
		opts.disable_function_signatures();
		
		if(type.isList() || type.isData() || type.isText()) {
			attributes[initializerName.cStr()] = methodDescriptor(py::cpp_function(
				[field](DynamicStructBuilder& builder, size_t n) {
					return builder.initList(field.getProto().getName(), n);
				},
				kj::str(
					initializerName,
					"(self: fusionsc.capnp.StructBuilder, size: n) -> ",
					fullName(field.getType())
				).cStr()
			));
		} else {
			attributes[initializerName.cStr()] = methodDescriptor(py::cpp_function(
				[field](DynamicStructBuilder& builder) {
					return builder.init(field.getProto().getName());
				},
				kj::str(
					initializerName,
					"(self: fusionsc.capnp.StructBuilder) -> ",
					fullName(field.getType())
				).cStr()
			));
		}
	};
	
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr rawName = field.getProto().getName();
		kj::String name = memberName(rawName);
		
		using Field = capnp::schema::Field;
		
		auto type = field.getType();		
		if(type.isList() || type.isData() || type.isText()) {
			addInitializer(field);
		}
		
		kj::String docString = kj::str("");
		KJ_IF_MAYBE(pSrc, maybeSource) {
			auto comment = pSrc -> getMembers()[field.getIndex()].getDocComment().asReader();
			// KJ_DBG("Field comment", structName, field.getProto().getName(), field.getIndex(), comment);
			if(comment.size() > 0) {
				docString = kj::heapString(comment);
			}
		}
		
		attributes[name.cStr()] = fieldDescriptor(*this, field, "Builder", docString);
	}
	
	for(StructSchema::Field field : schema.getUnionFields()) {
		auto type = field.getType();		
		
		if(type.isStruct()) {
			addInitializer(field);
		}
	}
	
	py::type baseClass = schema.isBranded() ? builderType(schema.getGeneric().asStruct()) : py::type::of<DynamicStructBuilder>();
	
	return makeType(
		*this,
		schema, "Builder",
		py::make_tuple(baseClass, this -> commonType(schema)),
		attributes
	);
}

py::type Loader::makePipelineType(capnp::StructSchema schema) {
	auto maybeSource = sourceInfo.find(schema.getProto().getId());
	
	py::dict attributes;
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr rawName = field.getProto().getName();
		kj::String name = memberName(rawName);
		
		using Field = capnp::schema::Field;
		
		auto type = field.getType();
		
		// Only emit pipeline fields for struct and interface fields
		if(!type.isStruct() && !type.isInterface())
			break;
		
		kj::String docString;
		KJ_IF_MAYBE(pSrc, maybeSource) {
			auto comment = pSrc -> getMembers()[field.getIndex()].getDocComment().asReader();
			// KJ_DBG("Field comment", structName, field.getProto().getName(), field.getIndex(), comment);
			if(comment.size() > 0) {
				docString = kj::heapString(comment);
			}
		}
		
		attributes[name.cStr()] = fieldDescriptor(*this, field, "Pipeline", docString);
	}
	
	py::type baseClass = schema.isBranded() ? pipelineType(schema.getGeneric().asStruct()) : py::type::of<DynamicStructPipeline>();
	
	return makeType(
		*this,
		schema, "Reader",
		py::make_tuple(baseClass),
		attributes
	);
}

py::type Loader::makeClientType(capnp::InterfaceSchema schema) {
	auto maybeSource = sourceInfo.find(schema.getProto().getId());
	
	py::dict attributes;
	
	auto methods = schema.getMethods();
	for(auto i : kj::indices(methods)) {
		auto pyMethod = makeInterfaceMethod(methods[i]);
		attributes[pyMethod.attr("__name__")] = pyMethod;
	}
	
	py::list basesList;
	for(auto baseType : schema.getSuperclasses()) {
		auto id = baseType.getProto().getId();
		basesList.append(clientType(baseType));
	}
	
	if(schema.isBranded()) {
		basesList.append(clientType(schema.getGeneric().asInterface()));
	}
	
	if(basesList.size() == 0) {
		basesList.append(py::type::of<DynamicCapabilityClient>());
	}
	
	return makeType(
		*this,
		schema, "Client",
		py::eval("tuple")(basesList),
		attributes
	);
}

py::type Loader::makeServerType(capnp::InterfaceSchema schema) {
	return makeType(
		*this,
		schema, "Server",
		py::make_tuple(py::type::of<DynamicCapabilityServer>()),
		py::dict()
	);
}

}
