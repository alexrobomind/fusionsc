#include <capnp/schema.capnp.h>
#include <capnp/serialize.h>
#include <capnp/serialize-text.h>
#include <capnp/any.h>

#include <kj/exception.h>
#include <kj/filesystem.h>
//#include <kj/miniposix.h>

#include <kj/main.h>

#include <stdexcept>

using capnp::schema::Node;
using capnp::schema::Field;
using capnp::schema::CodeGeneratorRequest;
using capnp::schema::Brand;
using capnp::schema::Type;
using capnp::schema::Value;

using kj::Array;
using kj::ArrayPtr;
using kj::ArrayBuilder;

using kj::String;
using kj::StringTree;

using kj::str;
using kj::strTree;

using kj::Maybe;

using kj::mv;

enum class NameUsage {
	RAW, USE_MEMBER
};

enum class ClassType {
	ROOT, BUILDER, READER
};

CodeGeneratorRequest::Reader request = CodeGeneratorRequest::Reader();
StringTree methodDefinitions = StringTree();

StringTree generateNested(uint64_t nodeId);

kj::String camelCase(kj::StringPtr in, bool firstUpper) {
	kj::Vector<char> result;
	
	bool upper = firstUpper;
	
	for(char c : in) {
		if(c >= 'a' && c <= 'z') {
			// Lower-case name
			if(upper)
				result.add(c - 'a' + 'A');
			else
				result.add(c);
			
			upper = false;
		} else if(c >= 'A' && c <= 'Z') {
			// Upper-case name
			result.add(c);
			
			upper = false;
		} else if(c == '_') {
			// Special characters
			upper = true;
		} else {
			result.add(c);
		}
		
	}
	
	result.add('\0');
	return kj::String(result.releaseAsArray());
}

kj::String enumCase(kj::StringPtr in) {
	kj::Vector<char> result;
	
	for(char c : in) {
		if(c >= 'a' && c <= 'z') {
			// Lower-case name
			result.add(c - 'a' + 'A');
		} else if(c >= 'A' && c <= 'Z') {
			// Upper-case name
			if(result.size() > 0)
				result.add('_');
			result.add(c);
		} else if(c == '_') {
			// Special characters
			result.add(c);
		} else {
			result.add(c);
		}
	}
	
	result.add('\0');
	return kj::String(result.releaseAsArray());
}

kj::String cppDefaultValue(Value::Reader val) {
	switch(val.which()) {
		case Value::VOID: return str("nullptr");
		case Value::BOOL: return str(val.getBool());
		case Value::INT8: return str(val.getInt8());
		case Value::INT16: return str(val.getInt16());
		case Value::INT32: return str(val.getInt32());
		case Value::INT64: return str(val.getInt64());
		case Value::UINT8: return str(val.getUint8());
		case Value::UINT16: return str(val.getUint16());
		case Value::UINT32: return str(val.getUint32());
		case Value::UINT64: return str(val.getUint64());
		case Value::FLOAT32: return str(val.getFloat32());
		case Value::FLOAT64: return str(val.getFloat64());
		case Value::ENUM: return str(val.getEnum());
		default: KJ_FAIL_REQUIRE("Unsupported default value type");
	}	
}

Node::Reader getNode(uint64_t id) {
	for(auto node : request.getNodes())
		if(node.getId() == id)
			return node;
	
	KJ_FAIL_REQUIRE("ID not found");
}

kj::Array<Node::Reader> fullScope(uint64_t id) {
	kj::Vector<Node::Reader> reverse;
	
	Node::Reader current = getNode(id);
	reverse.add(current);
	
	while(current.getScopeId() != 0) {
		current = getNode(current.getScopeId());
		reverse.add(current);
	}
	
	auto result = kj::heapArrayBuilder<Node::Reader>(reverse.size());
	while(reverse.size() != 0) {
		result.add(mv(reverse.back()));
		reverse.removeLast();
	}
	
	return result.finish();
}

kj::StringTree parameterName(uint64_t id, uint64_t index) {
	uint64_t counter = index;
	auto scope = fullScope(id);
	
	kj::StringPtr baseName = nullptr;
	
	for(auto node : scope) {
		if(node.getId() == id) {
			baseName = node.getParameters()[index].getName();
		} else {
			counter += node.getParameters().size();
		}
	}
	
	KJ_REQUIRE(baseName != nullptr);
	
	return strTree("Param", counter, baseName);
}

kj::StringPtr getNodeName(Node::Reader parent, Node::Reader child) {
	for(auto entry : parent.getNestedNodes()) {
		if(entry.getId() == child.getId())
			return entry.getName();
	}
	
	KJ_FAIL_REQUIRE("Nested node not found");
}

kj::StringTree cppTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, ClassType classType, NameUsage usage);
kj::StringTree cppNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, ClassType classType, NameUsage usage);

kj::StringTree cppTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, ClassType classType, NameUsage nameUsage) {
	StringTree result;
	
	switch(type.which()) {
		// Primitive types
		case Type::VOID: return strTree("nullptr_t");
		case Type::BOOL: return strTree("bool");
		case Type::INT8: return strTree("int8_t");
		case Type::INT16: return strTree("int16_t");
		case Type::INT32: return strTree("int32_t");
		case Type::INT64: return strTree("int64_t");
		case Type::UINT8: return strTree("uint8_t");
		case Type::UINT16: return strTree("uint16_t");
		case Type::UINT32: return strTree("uint32_t");
		case Type::UINT64: return strTree("uint64_t");
		case Type::FLOAT32: return strTree("float");
		case Type::FLOAT64: return strTree("double");
		
		case Type::TEXT: result = strTree("cupnp::Text"); break;
		case Type::DATA: result = strTree("cupnp::Data"); break;
		
		case Type::ENUM: {
			auto enumType = type.getEnum();
			return cppNodeTypeName(enumType.getTypeId(), enumType.getBrand(), scopeId, scopeBrand, classType, nameUsage);
		}		
		case Type::STRUCT: {
			auto structType = type.getStruct();
			return cppNodeTypeName(structType.getTypeId(), structType.getBrand(), scopeId, scopeBrand, classType, nameUsage);
		}	
		case Type::INTERFACE: {
			auto interfaceType = type.getInterface();
			return cppNodeTypeName(interfaceType.getTypeId(), interfaceType.getBrand(), scopeId, scopeBrand, classType, nameUsage);
		}
		
		case Type::LIST: {
			auto listType = type.getList();
			result = strTree("cupnp::List<", cppTypeName(listType.getElementType(), scopeId, scopeBrand, ClassType::ROOT, NameUsage::RAW), ">");
			break;
		}
		
		case Type::ANY_POINTER: {
			auto anyPointerType = type.getAnyPointer();
			switch(anyPointerType.which()) {
				case Type::AnyPointer::UNCONSTRAINED: {
					auto unconst = anyPointerType.getUnconstrained();
					switch(unconst.which()) {
						case Type::AnyPointer::Unconstrained::ANY_KIND: result = strTree("cupnp::AnyPointer"); break;
						case Type::AnyPointer::Unconstrained::STRUCT: result = strTree("cupnp::AnyStruct"); break;
						case Type::AnyPointer::Unconstrained::LIST: result = strTree("cupnp::AnyList"); break;
						case Type::AnyPointer::Unconstrained::CAPABILITY: result = strTree("cupnp::Capability"); break;
						KJ_FAIL_REQUIRE("Unknown unconstrained AnyPointer kind");
					}
					break;
				}
				case Type::AnyPointer::PARAMETER: {
					auto param = anyPointerType.getParameter();
					
					// We need to resolve the surrounding scope to find the right parameter name
					result = parameterName(param.getScopeId(), param.getParameterIndex());
	
					switch(classType) {
						case ClassType::BUILDER: return strTree("typename ", mv(result), "::Builder");
						case ClassType::READER: return strTree("typename ", mv(result), "::Reader");
						case ClassType::ROOT: return result;
					}
					KJ_UNREACHABLE;
				}
				case Type::AnyPointer::IMPLICIT_METHOD_PARAMETER: {
					KJ_FAIL_REQUIRE("Method parameters not supported");
				}
				KJ_FAIL_REQUIRE("Unknown AnyPointer kind");
			}
			break;
		}
		
		default:
			KJ_FAIL_REQUIRE("Unknown type kind");
	}
	
	switch(classType) {
		case ClassType::BUILDER: return strTree(mv(result), "::Builder");
		case ClassType::READER: return strTree(mv(result), "::Reader");
		case ClassType::ROOT: return result;
	}
	
	KJ_UNREACHABLE;
}

static constexpr uint64_t ANNOT_NAMESPACE = 0xb9c6f99ebf805f2cull;
static constexpr uint64_t ANNOT_NAME = 0xf264a779fef191ceull;

kj::StringTree nodeName(uint64_t nodeId) {
	kj::Array<Node::Reader> nodes = fullScope(nodeId);
	
	KJ_REQUIRE(nodes.size() > 1);
	
	size_t i = nodes.size() - 1;
	auto parent = nodes[i - 1];
	auto node   = nodes[i];
	
	StringTree nodeName;
	for(auto nested : parent.getNestedNodes()) {
		if(nested.getId() == node.getId())
			nodeName = strTree(nested.getName());
	}
	for(auto annot : node.getAnnotations()) {
		if(annot.getId() == ANNOT_NAME) {
			nodeName = strTree(annot.getValue().getText());
		}
	}
	if(parent.isStruct()) {
		auto structParent = parent.getStruct();
		
		for(auto field : structParent.getFields()) {
			if(!field.isGroup())
				continue;
			
			auto groupField = field.getGroup();
			
			if(groupField.getTypeId() == node.getId())
				nodeName = camelCase(field.getName(), true);
		}
	}
	
	return nodeName;
}

// Returns the C++ expression we have to use to refer to this node's generated type.
kj::StringTree cppNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, ClassType classType, NameUsage usage) {
	kj::Array<Node::Reader> scopeNodes = fullScope(scopeId);
	kj::Array<Node::Reader> nodes = fullScope(nodeId);
	
	// Enums have no readers or builders
	if(nodes[nodes.size() - 1].isEnum()/* && classType != ClassType::CAPNP_ROOT*/) {
		classType = ClassType::ROOT;
	}
	
	bool needsTypename = false;
	bool inDependentScope = false;
	
	StringTree result;
	
	// Check if there is a namespace annotation on the root	
	auto rootNode = nodes[0];
	
	if(rootNode.isFile()) {
		for(auto annot : rootNode.getAnnotations()) {
			if(annot.getId() == ANNOT_NAMESPACE) {
				result = strTree(annot.getValue().getText());
			}
		}
	}
	
	// if(usage != NameUsage::CAPNP_ROOT) {
		if(result.size() != 0)
			result = strTree(mv(result), "::cu");
		else
			result = strTree("cu");
	// }
	
	for(unsigned int i = 0; i < nodes.size(); ++i) {
		auto node = nodes[i];
		
		// Skip file nodes
		if(node.isFile()) {
			KJ_REQUIRE(i == 0);
			continue;
		}
		
		// KJ_REQUIRE(node.isStruct());
		
		if(inDependentScope)
			needsTypename = true;
		
		// Figure out this node's name
		KJ_REQUIRE(i >= 1);
		auto parent = nodes[i - 1];
		
		StringTree nodeName;
		for(auto nested : parent.getNestedNodes()) {
			if(nested.getId() == node.getId())
				nodeName = strTree(nested.getName());
		}
		for(auto annot : node.getAnnotations()) {
			if(annot.getId() == ANNOT_NAME) {
				nodeName = strTree(annot.getValue().getText());
			}
		}
		if(parent.isStruct()) {
			auto structParent = parent.getStruct();
			
			for(auto field : structParent.getFields()) {
				if(!field.isGroup())
					continue;
				
				auto groupField = field.getGroup();
				
				if(groupField.getTypeId() == node.getId())
					nodeName = camelCase(field.getName(), true);
			}
		}
		
		result = strTree(mv(result), "::", mv(nodeName));
		
		// If the node has no template parameter, we are done here
		if(node.getParameters().size() == 0) {
			continue;
		}
		
		bool scopeFound = false;
		
		auto inInherited = [&]() {
			// We are now in a dependent scope. If we have another nested name, we need a typename annotation
			inDependentScope = true;
			
			// Check that we are in a place where this is valid
			// Note: This is actually an invalid check because imported
			// nodes are not in the same global scope.
			/*bool foundInScope = false;
			for(auto scopeNode : scopeNodes)
				if(scopeNode.getId() == node.getId())
					foundInScope = true;
			KJ_REQUIRE(foundInScope, result);*/
			
			// Refer to type parameter for all templates
			auto nameBuilder = kj::heapArrayBuilder<kj::StringTree>(node.getParameters().size());
			for(unsigned int i = 0; i < node.getParameters().size(); ++i)
				nameBuilder.add(parameterName(node.getId(), i));
			
			result = strTree(mv(result), "<", StringTree(nameBuilder.finish(), ", "), ">");
		};
				
		// We need to figure out what to bind the template parameters to
		// Check the bindings to see whether there is a match
		for(auto scope : nodeBrand.getScopes()) {
			if(scope.getScopeId() != node.getId())
				continue;
			
			scopeFound = true;
			
			if(scope.isInherit()) {
				inInherited();
			} else if(scope.isBind()) {
				auto bind = scope.getBind();
				KJ_REQUIRE(bind.size() == node.getParameters().size());
				
				// Refer to bindings
				auto nameBuilder = kj::heapArrayBuilder<kj::StringTree>(node.getParameters().size());
				for(unsigned int i = 0; i < node.getParameters().size(); ++i) {
					KJ_REQUIRE(bind[i].isType());
					
					nameBuilder.add(cppTypeName(bind[i].getType(), scopeId, scopeBrand, ClassType::ROOT, NameUsage::RAW));
					
					// Having bindings can put us into a dependent scope that is hard to detect.
					needsTypename = true;
				}
				
				result = strTree(mv(result), "<", StringTree(nameBuilder.finish(), ", "), ">");
			}
		}
		
		if(!scopeFound) {
			// Assume we are in an inherited scope
			inInherited();
		}
	}
	
	if(inDependentScope) {
		switch(classType) {
			case ClassType::BUILDER:
			case ClassType::READER:
				needsTypename = true;
			
			case ClassType::ROOT:
			// case ClassType::CAPNP_ROOT:
				break;
		}
	}
	
	if(usage == NameUsage::USE_MEMBER)
		needsTypename = false;
			
	if(needsTypename)
		result = strTree("typename ", mv(result));
	
	switch(classType) {
		case ClassType::BUILDER:
			result = str(mv(result), "::Builder");
			break;
		
		case ClassType::READER:
			result = str(mv(result), "::Reader");
			break;
		
		case ClassType::ROOT:
		// case ClassType::CAPNP_ROOT:
			break;
	}
	
	return mv(result);
}

StringTree generateAllTemplateHeaders(uint64_t nodeId) {
	kj::Array<Node::Reader> nodes = fullScope(nodeId);
	
	StringTree result;
	
	for(auto node : nodes) {
		auto nParams = node.getParameters().size();
		
		kj::Vector<StringTree> paramNames;
		for(unsigned int i = 0; i < nParams; ++i) {
			paramNames.add(strTree("typename ", parameterName(node.getId(), i)));
		}
		
		if(paramNames.size() > 0) {
			result = strTree(
				mv(result),
				"template<", StringTree(paramNames.releaseAsArray(), ", "), ">\n"
			);
		}
	}
	
	return mv(result);
}

StringTree generateTemplateHeader(uint64_t nodeId) {
	kj::Array<Node::Reader> nodes = fullScope(nodeId);
	
	kj::Vector<StringTree> paramNames;
	
	KJ_REQUIRE(nodes.size() > 1);
	auto node = nodes[nodes.size() - 1];
	
	auto nParams = node.getParameters().size();
	
	if(nParams == 0)
		return strTree();
	
	for(unsigned int i = 0; i < nParams; ++i) {
		paramNames.add(strTree("typename ", parameterName(node.getId(), i)));
	}
	
	StringTree result = strTree(
		"template<", StringTree(paramNames.releaseAsArray(), ", "), ">\n"
	);
	
	return mv(result);
}

/* Checks whether we can specialize a template over this type. This is not possible if it is a child struct of a template */
bool canSpecializeOn(uint64_t nodeId) {
	kj::Array<Node::Reader> nodes = fullScope(nodeId);
	
	for(size_t i = 0; i < nodes.size(); ++i) {
		auto node = nodes[i];
		auto nParams = node.getParameters().size();
		
		if(i < nodes.size() - 1 && nParams != 0)
			return false;
	}
	
	return true;
}

StringTree generateInterface(uint64_t nodeId) {
	auto nodeBrand = capnp::defaultValue<Brand>();
	auto node = getNode(nodeId);
	
	KJ_REQUIRE(node.isInterface());
	auto asInterface = node.getInterface();
	
	auto name = nodeName(nodeId);
		
	StringTree result = strTree(
		generateTemplateHeader(nodeId),
		"struct ", name.flatten(), "{\n",
		"static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;\n",
		"\n",
		"// Interface pointer that holds the capability table offset\n",
		"uint64_t ptrData;\n",
		"\n",
		"inline CUPNP_FUNCTION ", name.flatten(), "(uint64_t structure, cupnp::Location data) : ptrData(structure) {\n",
		"	cupnp::validateInterfacePointer(structure, data);\n",
		"}\n",
		"\n",
		"inline CUPNP_FUNCTION bool isDefault() { return ptrData == 0; }\n",
		"\n",
		"friend CUPNP_FUNCTION void cupnp::swapData<", name.flatten(), ">(", name.flatten(), "&, ", name.flatten(), "&); \n",
		"\n",
		generateNested(nodeId),
		"};\n\n"
	);
	
	return result;
}

template<typename T>
StringTree generateValueAsBytes(typename T::Reader value) {
	// Use a message builder for type erasure
	capnp::MallocMessageBuilder builder1;
	builder1.setRoot(value);
	auto anyValue = builder1.getRoot<capnp::AnyPointer>().asReader();
	
	// Query size
	capnp::MallocMessageBuilder builder2(anyValue.targetSize().wordCount + 1);
	builder2.setRoot(anyValue);
	
	auto segments = builder2.getSegmentsForOutput();
	KJ_REQUIRE(segments.size() == 1);
	
	auto segment0 = segments[0];
	
	auto asBytes = segment0.asBytes();
	auto contents = kj::heapArrayBuilder<StringTree>(asBytes.size());
	for(unsigned char val : asBytes)
		contents.add(strTree(val));
	
	return strTree("{", StringTree(contents.finish(), ", "), "}");
}

StringTree generateValueAsBytes(Value::Reader value) {
	switch(value.which()) {
		case Value::TEXT: return generateValueAsBytes<capnp::Text>(value.getText());
		case Value::DATA: return generateValueAsBytes<capnp::Data>(value.getData());
		case Value::STRUCT: return generateValueAsBytes<capnp::AnyPointer>(value.getStruct());
		case Value::LIST: return generateValueAsBytes<capnp::AnyPointer>(value.getList());
		case Value::ANY_POINTER: return generateValueAsBytes<capnp::AnyPointer>(value.getAnyPointer());
		case Value::INTERFACE: return generateValueAsBytes<capnp::AnyPointer>(capnp::AnyPointer::Reader());
		
		default: KJ_FAIL_REQUIRE("Unknown value type", value);
	}
}

StringTree generateMethod(uint64_t nodeId, ClassType classType, StringTree returnType, StringTree name, StringTree contents) {
	auto nodeBrand = capnp::defaultValue<Brand>();
	kj::String typeName = cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, classType, NameUsage::USE_MEMBER).flatten();
	
	StringTree methodDeclaration = strTree(
		"inline CUPNP_FUNCTION ", returnType.flatten(), " ", name.flatten(), ";\n"
	);
	
	StringTree methodDefinition = strTree(
		generateAllTemplateHeaders(nodeId),
		"inline CUPNP_FUNCTION ", returnType.flatten(), " ", typeName, "::", name.flatten(), " { \n",
		mv(contents),
		"} \n\n"
	);
	
	methodDefinitions = strTree(
		mv(methodDefinitions),
		mv(methodDefinition)
	);
	
	return mv(methodDeclaration);
}

StringTree generateEnum(uint64_t nodeId) {
	auto nodeBrand = capnp::defaultValue<Brand>();
		
	auto node = getNode(nodeId);
	
	KJ_REQUIRE(node.isEnum());
	auto asEnum = node.getEnum();
	auto enumerants = asEnum.getEnumerants();
	
	auto name = nodeName(nodeId);
	
	StringTree result = strTree();
	
	for(size_t i = 0; i < enumerants.size(); ++i) {
		auto ecName = enumCase(enumerants[i].getName());
		result = strTree(
			mv(result),
			"    ", mv(ecName), " = ", i, ",\n"
		);
	}
	
	result = strTree(
		"enum class ", name.flatten(), " {\n",
		mv(result),
		"};\n\n"
	);
	
	return result;
}

StringTree generateStructSection(uint64_t nodeId, ClassType classType) {
	kj::StringPtr name = 
		classType == ClassType::BUILDER ?
			"Builder"_kj :
			"Reader"_kj
	;
	
	auto nodeBrand = capnp::defaultValue<Brand>();
		
	auto node = getNode(nodeId);
	KJ_REQUIRE(node.isStruct());
	auto asStruct = node.getStruct();
	
	auto fullName = cppNodeTypeName(nodeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), classType, NameUsage::USE_MEMBER);
	
	StringTree result = strTree(
		generateAllTemplateHeaders(nodeId),
		"struct ", fullName.flatten(), "{\n",
		"	\n",
		"	uint32_t dataSectionSize;\n",
		"	uint16_t pointerSectionSize;\n",
		"	cupnp::Location data;\n",
		"	\n",
		"	inline CUPNP_FUNCTION ", name, "(decltype(nullptr) nPtr) :\n",
		"		dataSectionSize(0),\n",
		"		pointerSectionSize(0),\n",
		"		data(nullptr)\n",
		"	{\n",
		"		cupnp::validateStructPointer(dataSectionSize, pointerSectionSize, data);\n",
		"	}\n",
		"	\n",
		"	inline CUPNP_FUNCTION ", name, "(uint64_t structure, cupnp::Location data) :\n",
		"		dataSectionSize(cupnp::getDataSectionSizeInBytes(structure)),\n",
		"		pointerSectionSize(cupnp::getPointerSectionSize(structure)),\n",
		"		data(data)\n",
		"	{\n",
		"		cupnp::validateStructPointer(dataSectionSize, pointerSectionSize, data);\n",
		"	}\n",
		"	\n",
		"	inline CUPNP_FUNCTION ", name, "(uint32_t dataSectionSize, uint16_t pointerSectionSize, cupnp::Location data) :\n",
		"		dataSectionSize(dataSectionSize),\n",
		"		pointerSectionSize(pointerSectionSize),\n",
		"		data(data)\n",
		"	{\n",
		"		cupnp::validateStructPointer(dataSectionSize, pointerSectionSize, data);\n",
		"	}\n",
		"	\n",
		"	inline CUPNP_FUNCTION bool isDefault() { return dataSectionSize == 0 && pointerSectionSize == 0; }\n",
		"	\n",
		"	friend CUPNP_FUNCTION void cupnp::swapData<", name, ">(", name, "&, ", name, "&); \n"
	);
	
	if(classType == ClassType::READER) {
		result = strTree(
			mv(result),
			generateMethod(
				nodeId, classType,
				strTree(""), strTree("Reader(const Builder& r)"),
				strTree(
					"dataSectionSize = r.dataSectionSize;\n",
					"pointerSectionSize = r.pointerSectionSize;\n",
					"data = r.data;\n"
				)
			)
		);
	}
	
	methodDefinitions = strTree(
		mv(methodDefinitions),
		"// ===== struct ", fullName.flatten(), " =====\n",
		"\n"
	);
	
	/*if(canSpecializeOn(nodeId)) {
		auto specializationTemplateHeader = generateAllTemplateHeaders(nodeId);
		
		if(specializationTemplateHeader.size() == 0)
			specializationTemplateHeader = strTree("template<>\n");
		
		auto capnpName = cppNodeTypeName(nodeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), true);
		
		methodDefinitions = strTree(
			mv(methodDefinitions),
			"// CuFor specializaation\n",
			"namespace cupnp {\n",
			mv(specializationTemplateHeader),
			"struct CuFor_<", capnpName.flatten(), "> { using Type = ", fullName.flatten(), "; }; \n",
			"} // namespace ::cupnp\n",
			"\n"
		);
	}*/
	
	for(auto field : asStruct.getFields()) {		
		result = strTree(mv(result), "\n// ", field.getName(), "\n");
		
		switch(field.which()) {
			case Field::GROUP: {
				auto typeId = field.getGroup().getTypeId();
				auto readerName = cppNodeTypeName(typeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), ClassType::BUILDER, NameUsage::RAW);
				auto builderName = cppNodeTypeName(typeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), ClassType::BUILDER, NameUsage::RAW);
				
				if(field.getDiscriminantValue() != 0xffff) {
					KJ_REQUIRE(asStruct.getDiscriminantCount() > 0);
					
					result = strTree(
						mv(result),
						generateMethod(
							nodeId, classType,
							strTree("bool"), strTree("is", camelCase(field.getName(), true), "() const"),
							strTree(
							"return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) == ", field.getDiscriminantValue(), ";\n"
							)
						),
						"	\n"
					);
				
					if(classType == ClassType::BUILDER) {
						result = strTree(
							mv(result),
							generateMethod(
								nodeId, classType,
								builderName.flatten(), strTree("mutate", camelCase(field.getName(), true), "()"),
								strTree(
								"cupnp::setDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data, ", field.getDiscriminantValue(), ");\n",
								"return ", builderName.flatten(), "(dataSectionSize, pointerSectionSize, data);\n"
								)
							)
						);
					}
				} else {
					if(classType == ClassType::BUILDER) {
						result = strTree(
							mv(result),
							generateMethod(
								nodeId, classType,
								builderName.flatten(), strTree("mutate", camelCase(field.getName(), true), "()"),
								strTree(
								"return ", builderName.flatten(), "(dataSectionSize, pointerSectionSize, data);\n"
								)
							)
						);
					}
				}
				
				result = strTree(
					mv(result),
					generateMethod(
						nodeId, classType,
						readerName.flatten(), strTree("get", camelCase(field.getName(), true), "() const"),
						strTree(
						"return ", readerName.flatten(), "(dataSectionSize, pointerSectionSize, data);\n"
						)
					),
					"\n"
				);
				break;
			}
			case Field::SLOT: {
				auto slot = field.getSlot();
				
				auto type = slot.getType();
				auto readerName = cppTypeName(type, nodeId, capnp::defaultValue<Brand>(), ClassType::READER, NameUsage::RAW);
				auto builderName = cppTypeName(type, nodeId, capnp::defaultValue<Brand>(), ClassType::BUILDER, NameUsage::RAW);
				auto typeName = cppTypeName(type, nodeId, capnp::defaultValue<Brand>(), ClassType::ROOT, NameUsage::RAW);
				
				auto fieldType = type.which() == Type::ENUM ? str("uint16_t") : typeName.flatten();
				
				auto subName = camelCase(field.getName(), true);
				auto enumName = enumCase(field.getName());
										
				auto castIfEnum = [&](kj::StringTree input, kj::String targetType) {
					if(type.which() != Type::ENUM)
						return mv(input);
					
					return strTree(
						"static_cast<", mv(targetType), ">(", mv(input), ")"
					);
				};
						
				kj::String accessorSuffix = type.isBool() ? 
					kj::str("BoolField<", slot.getOffset(), ">") :
					kj::str("PrimitiveField<", fieldType, ", ", slot.getOffset(), ">")
				;
				
				switch(type.which()) {
					case Type::ENUM:
					case Type::BOOL:
					case Type::INT8:
					case Type::INT16:
					case Type::INT32:
					case Type::INT64:
					case Type::UINT8:
					case Type::UINT16:
					case Type::UINT32:
					case Type::UINT64:
					case Type::FLOAT32:
					case Type::FLOAT64:
						if(field.getDiscriminantValue() != 0xffff) {
							// Getters and setters for unionized primitive fields
							result = strTree(
								mv(result),
								generateMethod(
									nodeId, classType,
									readerName.flatten(), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"if(cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) != ", field.getDiscriminantValue(), ") {\n",
										"	return ", castIfEnum(cppDefaultValue(slot.getDefaultValue()), readerName.flatten()), ";\n",
										"}\n",
										"return ", castIfEnum(strTree("cupnp::get", accessorSuffix, "(dataSectionSize, data, ", cppDefaultValue(slot.getDefaultValue()), ")"), readerName.flatten()), ";\n"
									)
								),
								generateMethod(
									nodeId, classType,
									strTree("void"), strTree("set", subName.asPtr(), "(", readerName.flatten(), " newVal)"),
									strTree(
										"cupnp::set", accessorSuffix, "(dataSectionSize, data, ", cppDefaultValue(slot.getDefaultValue()), ", ", castIfEnum(strTree("newVal"), str(fieldType)), ");\n",
										"cupnp::setDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data, ", field.getDiscriminantValue(), ");\n"
									)
								)
							);
						} else {
							// Getters and setters for non-unionized primitive fields
							result = strTree(
								mv(result),
								generateMethod(
									nodeId, classType,
									readerName.flatten(), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"return ", castIfEnum(strTree("cupnp::get", accessorSuffix, "(dataSectionSize, data, ", cppDefaultValue(slot.getDefaultValue()), ")"), readerName.flatten()), ";\n"
									)
								),
								generateMethod(
									nodeId, classType,
									strTree("void"), strTree("set", subName.asPtr(), "(", readerName.flatten(), " newVal)"),
									strTree(
										"cupnp::set", accessorSuffix, "(dataSectionSize, data, ", cppDefaultValue(slot.getDefaultValue()), ", ", castIfEnum(strTree("newVal"), str(fieldType)), ");\n"
									)
								)
							);
						}
						
						// NO BREAK HERE
					
					case Type::VOID:
					
						// Presence check for unionized fields
						if(field.getDiscriminantValue() != 0xffff) {
							KJ_REQUIRE(asStruct.getDiscriminantCount() > 0);
							
							result = strTree(
								mv(result),
								generateMethod(
									nodeId, classType,
									strTree("bool"), strTree("has", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) == ", field.getDiscriminantValue(), ";\n"
									)
								),
								generateMethod(
									nodeId, classType,
									strTree("void"), strTree("set", subName.asPtr(), "()"),
									strTree(
										"cupnp::setDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data, ", field.getDiscriminantValue(), ");\n"
									)
								)
							);
						}
						
						break;
					
					case Type::TEXT:
					case Type::DATA:
					case Type::LIST:
					case Type::STRUCT:
					case Type::ANY_POINTER: {						
						result = strTree(
							mv(result),
							"	inline static const unsigned char ", enumName.asPtr(), "_DEFAULT_VALUE [] = ", generateValueAsBytes(slot.getDefaultValue()), ";\n"
						);
							
						if(classType == ClassType::BUILDER) {
							result = strTree(
								mv(result),
								generateMethod(
									nodeId, classType,
									builderName.flatten(), strTree("mutate", subName.asPtr(), "()"),
									strTree(
										"CUPNP_REQUIRE(nonDefault", subName.asPtr(), "());\n",
										"return cupnp::mutatePointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(dataSectionSize, pointerSectionSize, data);\n"
									)
								)
							);
						}						
						
						if(field.getDiscriminantValue() != 0xffff) {
							KJ_REQUIRE(asStruct.getDiscriminantCount() > 0);
							
							result = strTree(
								mv(result),
							
								generateMethod(
									nodeId, classType,
									strTree(readerName.flatten()), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"if(cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) != ", field.getDiscriminantValue(), ") {\n",
										"	return cupnp::getPointer<", readerName.flatten(), ">(reinterpret_cast<const capnp::word*>(", enumName.asPtr(), "_DEFAULT_VALUE));\n",
										"} \n",
										"return cupnp::getPointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(dataSectionSize, pointerSectionSize, data, reinterpret_cast<const capnp::word*>(", enumName.asPtr(), "_DEFAULT_VALUE));\n"
									)
								),
								generateMethod(
									nodeId, classType,
									strTree("bool"), strTree("nonDefault", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) == ", field.getDiscriminantValue(), " && cupnp::hasPointerField<", slot.getOffset(), ">(dataSectionSize, pointerSectionSize, data);\n"
									)
								),
								generateMethod(
									nodeId, classType,
									strTree("bool"), strTree("has", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) == ", field.getDiscriminantValue(), ";\n"
									)
								)
							);
						} else {
							result = strTree(
								mv(result),
								
								generateMethod(
									nodeId, classType,
									strTree(readerName.flatten()), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::getPointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(dataSectionSize, pointerSectionSize, data, reinterpret_cast<const capnp::word*>(", enumName.asPtr(), "_DEFAULT_VALUE));\n"
									)
								),
								generateMethod(
									nodeId, classType,
									strTree("bool"), strTree("nonDefault", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::hasPointerField<", slot.getOffset(), ">(dataSectionSize, pointerSectionSize, data);\n"
									)
								)
							);
						}
						break;
					}
					
					case Type::INTERFACE:						
						if(field.getDiscriminantValue() != 0xffff) {
							KJ_REQUIRE(asStruct.getDiscriminantCount() > 0);
							
							result = strTree(
								mv(result),
								
								generateMethod(
									nodeId, classType,
									strTree("bool"), strTree("nonDefault", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) == ", field.getDiscriminantValue(), " && cupnp::hasPointerField<", slot.getOffset(), ">(dataSectionSize, pointerSectionSize, data);\n"
									)
								),
								generateMethod(
									nodeId, classType,
									strTree("bool"), strTree("has", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(dataSectionSize, data) == ", field.getDiscriminantValue(), ";\n"
									)
								)
							);
						} else {
							result = strTree(
								mv(result),
								
								generateMethod(
									nodeId, classType,
									strTree("bool"), strTree("nonDefault", subName.asPtr(), "() const"),
									strTree(
										"return cupnp::hasPointerField<", slot.getOffset(), ">(dataSectionSize, pointerSectionSize, data);\n"
									)
								)
							);
						}
						break;
				}
				break;
			}
		}
	}
	
	result = strTree(
		mv(result), 	
		"}; // struct ", name, "\n"
	);
	
	return mv(result);
	
}

StringTree generateStruct(uint64_t nodeId) {
	StringTree builderDef = generateStructSection(nodeId, ClassType::BUILDER);
	StringTree readerDef  = generateStructSection(nodeId, ClassType::READER);
	
	methodDefinitions = strTree(
		mv(builderDef), "\n",
		mv(readerDef), "\n",
		mv(methodDefinitions)
	);
	
	return strTree(
		generateTemplateHeader(nodeId),
		"struct ", nodeName(nodeId), " {\n",
		"static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;\n",
		"struct Reader;\n",
		"struct Builder;\n",
		"\n",
		generateNested(nodeId),
		"};\n"
	);
}

StringTree generateNode(uint64_t nodeId) {
	StringTree result;
	
	auto node = getNode(nodeId);
	
	if(node.isStruct()) {		
		result = strTree(mv(result), generateStruct(node.getId()));
	}
	
	if(node.isInterface()){
		result = strTree(mv(result), generateInterface(node.getId()));
	}
	
	if(node.isEnum()) {
		result = strTree(mv(result), generateEnum(node.getId()));
	}
	
	return mv(result);
}

StringTree generateNested(uint64_t nodeId) {
	StringTree result;
	
	auto node = getNode(nodeId);
	
	if(node.isStruct()) {
		auto structNode = node.getStruct();
		
		for(auto field : structNode.getFields()) {
			if(!field.isGroup())
				continue;
			
			auto groupField = field.getGroup();
			result = strTree(mv(result), generateNode(groupField.getTypeId()));
		}
	}
	
	for(auto child : node.getNestedNodes()) {
		result = strTree(mv(result), generateNode(child.getId()));
	}
	
	return mv(result);
}

StringTree generateForwardDeclarations(kj::String namespaceName, uint64_t nodeId) {
	auto fileNode = getNode(nodeId);
	
	StringTree result;
	for(auto child : fileNode.getNestedNodes()) {
		uint64_t childId = child.getId();
		
		auto childNode = getNode(childId);
		
		if(childNode.isStruct() || childNode.isInterface()) {
			result = strTree(
				mv(result), "\n",
				generateAllTemplateHeaders(childId),
				"struct ", nodeName(childId).flatten(), ";"
			);
		}
		
		if(childNode.isEnum()) {
			result = strTree(
				mv(result), "\n",
				generateAllTemplateHeaders(childId),
				"enum class ", nodeName(childId).flatten(), ";"
			);
		}
	}
	
	return result;
}

StringTree generateKindOverrides(kj::String namespaceName, uint64_t nodeId) {
	auto fileNode = getNode(nodeId);
	
	StringTree result;
	for(auto child : fileNode.getNestedNodes()) {
		uint64_t childId = child.getId();
		
		auto childNode = getNode(childId);
		
		if(childNode.isStruct()) {
			auto specializationTemplateHeader = generateAllTemplateHeaders(childId);
			
			if(specializationTemplateHeader.size() == 0)
				specializationTemplateHeader = strTree("template<>\n");
			
			result = strTree(
				mv(result), "\n",
				mv(specializationTemplateHeader),
				"struct KindFor_<", cppNodeTypeName(childId, capnp::defaultValue<Brand>(), childId, capnp::defaultValue<Brand>(), ClassType::ROOT, NameUsage::RAW).flatten(), "> { static inline constexpr Kind kind = Kind::STRUCT; };"
			);
		}
		
		if(childNode.isInterface()) {
			auto specializationTemplateHeader = generateAllTemplateHeaders(childId);
			
			if(specializationTemplateHeader.size() == 0)
				specializationTemplateHeader = strTree("template<>\n");
			
			result = strTree(
				mv(result), "\n",
				mv(specializationTemplateHeader),
				"struct KindFor_<", cppNodeTypeName(childId, capnp::defaultValue<Brand>(), childId, capnp::defaultValue<Brand>(), ClassType::ROOT, NameUsage::RAW).flatten(), "> { static inline constexpr Kind kind = Kind::INTERFACE; };"
			);
		}
		
		if(childNode.isEnum()) {
			auto specializationTemplateHeader = generateAllTemplateHeaders(childId);
			
			if(specializationTemplateHeader.size() == 0)
				specializationTemplateHeader = strTree("template<>\n");
			
			result = strTree(
				mv(result), "\n",
				mv(specializationTemplateHeader),
				"struct KindFor_<", cppNodeTypeName(childId, capnp::defaultValue<Brand>(), childId, capnp::defaultValue<Brand>(), ClassType::ROOT, NameUsage::RAW).flatten(), "> { static inline constexpr Kind kind = Kind::ENUM; };"
			);
		}
	}
	
	return result;
}

void generateRequested() {
	auto fs = kj::newDiskFilesystem();
	auto& cwd = fs->getCurrent();
	
	for(auto fileNode : request.getRequestedFiles()) {
		StringTree namespaceName;
		
		//if(rootNode.isFile()) {
		auto node = getNode(fileNode.getId());
		for(auto annot : node.getAnnotations()) {
			if(annot.getId() == ANNOT_NAMESPACE) {
				namespaceName = strTree(annot.getValue().getText());
			}
		}
		//}
		
		if(namespaceName.size() != 0)
			namespaceName = strTree(mv(namespaceName), "::cu");
		else
			namespaceName = strTree("cu");
		
		StringTree forwardDeclarations = generateForwardDeclarations(namespaceName.flatten(), fileNode.getId());
		StringTree kindOverrides = generateKindOverrides(namespaceName.flatten(), fileNode.getId());
		
		StringTree declarations = generateNested(fileNode.getId());
		
		kj::String flatNsName = namespaceName.flatten();
		auto openNS = strTree();
		auto closeNS = strTree();
		
		{
			kj::StringPtr nsSubName = flatNsName;
		
			while(true) {
				KJ_IF_MAYBE(idx, nsSubName.findFirst(':')) {
					KJ_REQUIRE(nsSubName[*idx+1] == ':');
					
					openNS = strTree(mv(openNS), "namespace ", nsSubName.slice(0, *idx), "{\n");
					closeNS = strTree(mv(closeNS), "}");
					
					nsSubName = nsSubName.slice(*idx + 2);
				} else {
					openNS = strTree(mv(openNS), "namespace ", nsSubName, "{\n");
					closeNS = strTree(mv(closeNS), "}");
					
					break;
				}
			}
		}
		
		auto inputFilename = fileNode.getFilename();
		kj::Path inputFile = kj::Path::parse(inputFilename);
		
		StringTree result = strTree(
			"#pragma once \n",
			"\n",
			"#include <cupnp/cupnp.h>\n",
			"#include \"", inputFilename, ".h\"\n",
			KJ_MAP(importNode, fileNode.getImports()) {				
				auto path = importNode.getName();
								
				// We provide capnp classes ourselves
				if(path.startsWith("/capnp/"))
					return strTree();
				
				if(path.startsWith("/"))
					return strTree("#include<", path.slice(1), ".cu.h>\n");
				else
					return strTree("#include \"", path, ".cu.h\"\n");
			},
			"\n",
			//"namespace ", namespaceName.flatten(), " {\n",
			openNS.flatten(), "\n",
			"\n",
			"// --- Forward declarations ---\n",
			"\n",
			mv(forwardDeclarations),
			"\n",
			closeNS.flatten(), "\n",
			"\n",
			"namespace cupnp { namespace internal {\n",
			"\n",
			"// --- Kind overrides ---\n",
			"\n",
			mv(kindOverrides),
			"\n",
			"}}\n",
			"\n",
			openNS.flatten(), "\n",
			"\n",
			"// --- Declarations ---\n",
			"\n",
			mv(declarations),
			"\n",
			closeNS.flatten(), "\n",
			"\n",
			"// --- Method definitions ---\n",
			"\n",
			mv(methodDefinitions)
		);
		
		/*
		// Format output
		// WARNING: This is simple because we don't create string literals
		// If they get ever added, we need to change the formatting code to handle them
		kj::StringTree formatted;
		kj::Vector<char> lineBuffer;
		
		result = strTree(mv(result), "\n");
		
		uint32_t indent = 0;
		uint32_t indentNext = 0;
		bool inWs = false;
		
		auto flat = result.flatten();
		for(char c : flat) {
			if(c == ' ' || c == '\t') {
				// Turn non-linebreak whitespaces into spaces
				if(inWs) {
					continue;
				} else {
					lineBuffer.add(' ');
					inWs = true;
				}
			} else if(c == '\n') {
				// Linebreaks get indentation
				lineBuffer.add('\0');
				StringTree formattedLine = strTree(heapString(lineBuffer));
				lineBuffer.clear();
				
				for(auto i : kj::range(0, indent))
					formattedLine = strTree("    ", mv(formattedLine));
				
				KJ_DBG(formattedLine);
				
				formatted = strTree(mv(formatted), "\n", mv(formattedLine));
				inWs = true;
				indent += indentNext;
				indentNext = 0;
			} else {
				// Non-whitespace chars get passed through
				inWs = false;
				lineBuffer.add(c);
				
				if(c == '{')
					++indentNext;
				
				if(c == '}') {
					if(indentNext > 0) {
						--indentNext;
					} else {
						KJ_REQUIRE(indent > 0, "Generated '}' without matching '{'");
						--indent;
					}
				}
			}
		}
		
		KJ_REQUIRE(indent == 0, "Not all '{' were closed by '}'");
		*/
		
		/*if(baseName.size() < 6 || baseName.slice(baseName.size() - 6) != ".capnp") {
			KJ_LOG(WARNING, "Skipped file because its name does not end with '.capnp'", baseName);
			continue;
		}*/
		
		//kj::Vector<char> baseNameVec;
		//baseNameVec.addAll(inputFile.basename()[0]);
		//baseNameVec.resize(baseNameVec.size() - 6);
		//baseNameVec.add('\0');
		
		//kj::String baseName(baseNameVec.releaseAsArray());
		//KJ_LOG(WARNING, baseName);
		
		kj::String headerName = str(inputFilename, ".cu.h"); // str(baseName, ".cupnp.h");
		
		auto outFile = cwd.openFile(kj::Path::parse(headerName), kj::WriteMode::CREATE | kj::WriteMode::MODIFY | kj::WriteMode::CREATE_PARENT );
		outFile -> writeAll(result.flatten());
	}
	
	// return mv(result);
}

void mainFunc(kj::StringPtr programName, kj::ArrayPtr<const kj::StringPtr> args) {
	//KJ_LOG(WARNING, "Initiating compilation");
    capnp::ReaderOptions options;
    options.traversalLimitInWords = 1 << 30;	
	
	capnp::StreamFdMessageReader input(0, options);
	request = input.getRoot<CodeGeneratorRequest>();
	
	// Format input
	/*KJ_LOG(WARNING, "Formatting input");
	capnp::TextCodec outputCodec;
	outputCodec.setPrettyPrint(true);
	kj::String result = outputCodec.encode(root);
	KJ_LOG(WARNING, result);*/
	
	generateRequested();
	
	// Open file
	/*auto fs = kj::newDiskFilesystem();
	auto& cwd = fs->getCurrent();
	
	auto file = cwd.openFile(kj::Path("output.txt"), kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
	
	// Write formatted output
	file->writeAll(result);
	
	auto file2 = cwd.openFile(kj::Path("output2.txt"), kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
	file2 -> writeAll(str(generateRequested(root)));*/
}

int main(int argc, char** argv) {
	// Note: This roundabout is neccessary, as KJs runMainAndExit reconfigures the std input and output to binary if not run on console
	// This is neccessary to properly communicate with the capnp compiler that hands us compilation requests
	kj::TopLevelProcessContext ctx("cupnpc");
	kj::runMainAndExit(ctx, mainFunc, argc, argv);
	return 0;
}