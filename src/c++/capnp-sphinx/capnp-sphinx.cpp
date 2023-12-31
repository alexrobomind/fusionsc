#include <capnp/schema.capnp.h>
#include <capnp/serialize.h>
#include <capnp/serialize-text.h>
#include <capnp/any.h>

#include <kj/exception.h>
#include <kj/filesystem.h>
//#include <kj/miniposix.h>

#include <kj/main.h>

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
using kj::StringPtr;

using kj::str;
using kj::strTree;

using kj::Maybe;

using kj::mv;



StringTree generateNested(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringTree& methodDefinitions);

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

Node::Reader getNode(CodeGeneratorRequest::Reader request, uint64_t id) {
	for(auto node : request.getNodes())
		if(node.getId() == id)
			return node;
	
	KJ_FAIL_REQUIRE("ID not found");
}

Node::SourceInfo::Reader getSourceInfo(CodeGeneratorRequest::Reader request, uint64_t id) {
	for(auto srcInfo : request.getSourceInfo())
		if(srcInfo.getId() == id)
			return srcInfo;
	
	KJ_FAIL_REQUIRE("ID not found");
}
	

kj::Array<Node::Reader> fullScope(CodeGeneratorRequest::Reader request, uint64_t id) {
	kj::Vector<Node::Reader> reverse;
	
	Node::Reader current = getNode(request, id);
	reverse.add(current);
	
	while(current.getScopeId() != 0) {
		current = getNode(request, current.getScopeId());
		reverse.add(current);
	}
	
	auto result = kj::heapArrayBuilder<Node::Reader>(reverse.size());
	while(reverse.size() != 0) {
		result.add(mv(reverse.back()));
		reverse.removeLast();
	}
	
	return result.finish();
}

kj::StringTree parameterName(CodeGeneratorRequest::Reader request, uint64_t id, uint64_t index) {
	uint64_t counter = index;
	auto scope = fullScope(request, id);
	
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

kj::StringTree cppTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request, bool capnpType = false);
kj::StringTree pythonTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request);
kj::StringTree cppNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request, bool capnpType = false);
kj::StringTree pythonNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request);

kj::StringTree pythonTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request, bool capnpType, bool builder) {
	auto suffix = builder ? "Builder" : "Reader";
	
	switch(type.which()) {
		// Primitive types
		case Type::VOID: return strTree("NoneType");
		case Type::BOOL: return strTree("bool");
		case Type::INT8: return strTree("int");
		case Type::INT16: return strTree("int");
		case Type::INT32: return strTree("int");
		case Type::INT64: return strTree("int");
		case Type::UINT8: return strTree("int");
		case Type::UINT16: return strTree("int");
		case Type::UINT32: return strTree("int");
		case Type::UINT64: return strTree("int");
		case Type::FLOAT32: return strTree("float");
		case Type::FLOAT64: return strTree("float");
		case Type::TEXT: return strTree("fsc.capnp.Text.", suffix);
		case Type::DATA: return strTree("fsc.capnp.Data.", suffix);
		
		case Type::LIST: {
			auto listType = type.getList();
			auto subType = pythonTypeName(listType.getElementType(), scopeId, scopeBrand, request);
			return strTree("fsc.capnp.List[", mv(subType), "].", suffix);
		}
		
		case Type::ENUM: {
			auto enumType = type.getEnum();
			return pythonNodeTypeName(enumType.getTypeId(), enumType.getBrand(), scopeId, scopeBrand, request);
		}		
		case Type::STRUCT: {
			auto structType = type.getStruct();
			return pythonNodeTypeName(structType.getTypeId(), structType.getBrand(), scopeId, scopeBrand, request);
		}	
		case Type::INTERFACE: {
			auto interfaceType = type.getInterface();
			return pythonNodeTypeName(interfaceType.getTypeId(), interfaceType.getBrand(), scopeId, scopeBrand, request);
		}
		
		case Type::ANY_POINTER: {
			auto anyPointerType = type.getAnyPointer();
			switch(anyPointerType.which()) {
				case Type::AnyPointer::UNCONSTRAINED: {
					auto unconst = anyPointerType.getUnconstrained();
					switch(unconst.which()) {
						case Type::AnyPointer::Unconstrained::ANY_KIND: return strTree("fsc.cupnp.AnyPointer.", suffix);
						case Type::AnyPointer::Unconstrained::STRUCT: return strTree("fsc.cupnp.AnyStruct.", suffix);
						case Type::AnyPointer::Unconstrained::LIST: return strTree("fsc.cupnp.AnyList.", suffix);
						case Type::AnyPointer::Unconstrained::CAPABILITY: return strTree("fsc.cupnp.Capability.", suffix);
						KJ_FAIL_REQUIRE("Unknown unconstrained AnyPointer kind");
					}
				}
				case Type::AnyPointer::PARAMETER: {
					auto param = anyPointerType.getParameter();
					
					// We need to resolve the surrounding scope to find the right parameter name
					return parameterName(request, param.getScopeId(), param.getParameterIndex());
				}
				case Type::AnyPointer::IMPLICIT_METHOD_PARAMETER: {
					KJ_FAIL_REQUIRE("Method parameters not supported");
				}
				KJ_FAIL_REQUIRE("Unknown AnyPointer kind");
			}
		}
		
		default:
			KJ_FAIL_REQUIRE("Unknown type kind");
	}
}


kj::StringTree cppTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request, bool capnpType) {
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
		case Type::TEXT: return strTree(capnpType ? "capnp::Text" : "cupnp::Text");
		case Type::DATA: return strTree(capnpType ? "capnp::Data" : "cupnp::Data");
		
		case Type::LIST: {
			auto listType = type.getList();
			return strTree("cupnp::List<", cppTypeName(listType.getElementType(), scopeId, scopeBrand, request), ">");
		}
		
		case Type::ENUM: {
			auto enumType = type.getEnum();
			return cppNodeTypeName(enumType.getTypeId(), enumType.getBrand(), scopeId, scopeBrand, request);
		}		
		case Type::STRUCT: {
			auto structType = type.getStruct();
			return cppNodeTypeName(structType.getTypeId(), structType.getBrand(), scopeId, scopeBrand, request);
		}	
		case Type::INTERFACE: {
			auto interfaceType = type.getInterface();
			return cppNodeTypeName(interfaceType.getTypeId(), interfaceType.getBrand(), scopeId, scopeBrand, request);
		}
		
		case Type::ANY_POINTER: {
			auto anyPointerType = type.getAnyPointer();
			switch(anyPointerType.which()) {
				case Type::AnyPointer::UNCONSTRAINED: {
					auto unconst = anyPointerType.getUnconstrained();
					switch(unconst.which()) {
						case Type::AnyPointer::Unconstrained::ANY_KIND: return strTree("cupnp::AnyPointer");
						case Type::AnyPointer::Unconstrained::STRUCT: return strTree("cupnp::AnyStruct");
						case Type::AnyPointer::Unconstrained::LIST: return strTree("cupnp::AnyList");
						case Type::AnyPointer::Unconstrained::CAPABILITY: return strTree("cupnp::Capability");
						KJ_FAIL_REQUIRE("Unknown unconstrained AnyPointer kind");
					}
				}
				case Type::AnyPointer::PARAMETER: {
					auto param = anyPointerType.getParameter();
					
					// We need to resolve the surrounding scope to find the right parameter name
					return parameterName(request, param.getScopeId(), param.getParameterIndex());
				}
				case Type::AnyPointer::IMPLICIT_METHOD_PARAMETER: {
					KJ_FAIL_REQUIRE("Method parameters not supported");
				}
				KJ_FAIL_REQUIRE("Unknown AnyPointer kind");
			}
		}
		
		default:
			KJ_FAIL_REQUIRE("Unknown type kind");
	}
}

static constexpr uint64_t ANNOT_NAMESPACE = 0xb9c6f99ebf805f2cull;
static constexpr uint64_t ANNOT_NAME = 0xf264a779fef191ceull;

kj::StringTree nodeName(uint64_t nodeId, CodeGeneratorRequest::Reader request) {
	kj::Array<Node::Reader> nodes = fullScope(request, nodeId);
	
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

kj::StringTree pythonNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request, bool capnpType) {
	kj::Array<Node::Reader> scopeNodes = fullScope(request, scopeId);
	kj::Array<Node::Reader> nodes = fullScope(request, nodeId);
	
	StringTree result;
	
	for(unsigned int i = 0; i < nodes.size(); ++i) {
		auto node = nodes[i];
		
		// Skip file nodes
		if(node.isFile()) {
			KJ_REQUIRE(i == 0);
			continue;
		}
		
		result = strTree(mv(result), ".", mv(nodeName));
	}
	
	return result;
}
	
//! Warning: In constrast to its cupnpc equivalent, this does not put "typename" in where C++ requires it
kj::StringTree cppNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request, bool capnpType) {
	kj::Array<Node::Reader> scopeNodes = fullScope(request, scopeId);
	kj::Array<Node::Reader> nodes = fullScope(request, nodeId);
	
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
	
	if(!capnpType) {
		if(result.size() != 0)
			result = strTree(mv(result), "::cu");
		else
			result = strTree("cu");
	}
	
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
			bool foundInScope = false;
			for(auto scopeNode : scopeNodes)
				if(scopeNode.getId() == node.getId())
					foundInScope = true;
			KJ_REQUIRE(foundInScope);
			
			// Refer to type parameter for all templates
			auto nameBuilder = kj::heapArrayBuilder<kj::StringTree>(node.getParameters().size());
			for(unsigned int i = 0; i < node.getParameters().size(); ++i)
				nameBuilder.add(parameterName(request, node.getId(), i));
			
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
					
					nameBuilder.add(cppTypeName(bind[i].getType(), scopeId, scopeBrand, request));
				}
				
				result = strTree(mv(result), "<", StringTree(nameBuilder.finish(), ", "), ">");
			}
		}
		
		if(!scopeFound) {
			// Assume we are in an inherited scope
			inInherited();
		}
	}
	
	/*if(needsTypename)
		result = strTree("typename ", mv(result));*/
	
	return mv(result);
}

// WARNING: This function differs from its cupnpc counterpart, as sphinx requires a backslash before the linebreak
StringTree generateAllTemplateHeaders(CodeGeneratorRequest::Reader request, uint64_t nodeId) {
	kj::Array<Node::Reader> nodes = fullScope(request, nodeId);
	
	StringTree result;
	
	for(auto node : nodes) {
		auto nParams = node.getParameters().size();
		
		kj::Vector<StringTree> paramNames;
		for(unsigned int i = 0; i < nParams; ++i) {
			paramNames.add(strTree("typename ", parameterName(request, node.getId(), i)));
		}
		
		if(paramNames.size() > 0) {
			result = strTree(
				mv(result),
				"template<", StringTree(paramNames.releaseAsArray(), ", "), ">\\\n"
			);
		}
	}
	
	return mv(result);
}

// WARNING: This function differs from its cupnpc counterpart, as sphinx requires a backslash before the linebreak
StringTree generateTemplateHeader(CodeGeneratorRequest::Reader request, uint64_t nodeId) {
	kj::Array<Node::Reader> nodes = fullScope(request, nodeId);
	
	kj::Vector<StringTree> paramNames;
	
	KJ_REQUIRE(nodes.size() > 1);
	auto node = nodes[nodes.size() - 1];
	
	auto nParams = node.getParameters().size();
	
	if(nParams == 0)
		return strTree();
	
	for(unsigned int i = 0; i < nParams; ++i) {
		paramNames.add(strTree("typename ", parameterName(request, node.getId(), i)));
	}
	
	StringTree result = strTree(
		"template<", StringTree(paramNames.releaseAsArray(), ", "), ">\\\n"
	);
	
	return mv(result);
}

StringTree indent(StringPtr code, StringPtr prefix) {
	StringTree result;
	
	while(true) {
		KJ_IF_MAYBE(pLineBreak, code.findFirst('\n')) {
			result = strTree(mv(result), prefix, code.slice(0, *pLineBreak), "\n");
			code = code.slice(*pLineBreak + 1);
		} else {
			result = strTree(mv(result), prefix, code);
			break;
		}
	}
	
	return result;
}

StringTree generateInterfaceDocumentation(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringPtr indentPrefix) {
	auto nodeBrand = capnp::defaultValue<Brand>();
	auto node = getNode(request, nodeId);
	
	KJ_REQUIRE(node.isInterface());
	auto asInterface = node.getInterface();
	
	auto name = nodeName(nodeId, request);
	kj::String cppName = cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request).flatten();
	kj::String pyName = pythonNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request).flatten();
	
	auto sourceInfo = getSourceInfo(request, nodeId);
	
	StringTree result = strTree(
		indentPrefix, ".. cpp:struct ", generateAllTemplateHeaders(request, nodeId), name.flatten(), cppName, "::\n",
		indent(sourceInfo.getDocComment(), str(indentPrefix, "  ")),
		"\n",
		
		indentPrefix, ".. py:struct ", pyName, "::\n",
		indent(sourceInfo.getDocComment(), str(indentPrefix, "  ")),
		"\n"
	);
	
	return result;
}

StringTree generateStructDocumentation(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringPtr indentPrefix) {
	auto nodeBrand = capnp::defaultValue<Brand>();
	auto node = getNode(request, nodeId);
	
	KJ_REQUIRE(node.isInterface());
	auto asInterface = node.getInterface();
	
	auto name = nodeName(nodeId, request);
	
	StringTree result;
	
	auto sourceInfo = getSourceInfo(request, nodeId);
	
	for(auto i : kj::range(0, 2)) {
		bool builder = i == 0;
		
		kj::String cppName = cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request, builder).flatten();
		kj::String pyName = pythonNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request, builder).flatten();
		
		result = strTree(
			mv(result),
			
			indentPrefix, ".. cpp:struct ", generateAllTemplateHeaders(request, nodeId), name.flatten(), cppName, "::\n",
			indent(sourceInfo.getDocComment(), str(indentPrefix, "  ")),
			"\n",
			
			indentPrefix, ".. py:struct ", pyName, "::\n",
			indent(sourceInfo.getDocComment(), str(indentPrefix, "  ")),
			"\n"
		);
	}
	
	return result;
}

StringTree generateMethodDocumentation(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringTree& methodDefinitions, StringTree returnType, StringTree name, StringTree contents, StringPtr indentPrefix) {
	auto nodeBrand = capnp::defaultValue<Brand>();
	
	kj::String cppName = cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request).flatten();
	kj::String pyName = pythonNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request).flatten();
	
	auto sourceInfo = getSourceInfo(request, nodeId);
	
	StringTree methodDocumentation = strTree(
		indentPrefix, ".. cpp:function ",
		generateAllTemplateHeaders(request, nodeId),
		returnType.flatten(), " ", cppName, "::", name.flatten(), "::\n",
		indent(sourceInfo.getDocComment(), str(indentPrefix, "  ")),
		
		indentPrefix, ".. py:method ", pyName, ".", name.flatten(), "::\n",
		indent(sourceInfo.getDocComment(), str(indentPrefix, "  "))
	);
	
	return mv(methodDocumentation);
}

StringTree generateFieldDocumentation(CodeGeneratorRequest::Reader request, uint64_t nodeId, Field::Reader field, capnp::schema::Node::SourceInfo::Reader fieldInfo, StringPtr indentPrefix) {	
	auto nodeBrand = capnp::defaultValue<Brand>();
	auto node = getNode(request, nodeId);
	
	KJ_REQUIRE(node.isStruct());
	auto asInterface = node.getStruct();
	
	auto fieldName = field.getName();
	
	StringTree result;
	
	kj::String cppType = cppTypeName(field.getType());
	
	for(auto i : kj::range(0, 2)) {
		bool builder = i == 0;
		
		kj::String cppTN = cppTypeName(field.getType(), nodeId, capnp::defaultValue<Brand>(), request, builder).flatten();
		kj::String pyTN = pythonTypeName(field.getType(), nodeId, capnp::defaultValue<Brand>(), request, builder).flatten();
		kj::String cppName = cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request, builder).flatten();
		kj::String pyName = pythonNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request, builder).flatten();
		
		result = strTree(
			mv(result),
			
			indentPrefix, ".. cpp:member ", generateAllTemplateHeaders(request, nodeId), cppTN, " ", cppName, "::", fieldName, "::\n",
			indent(fieldInfo.getDocComment(), str(indentPredix, "  ")),
			"\n",
			
			indentPrefix, ".. py:attribute ", pyTN, " ", pyName, ".", fieldName, "::\n",
			indent(fieldInfo.getDocComment(), str(indentPredix, "  ")),
			"\n",
		);
	}
	
	return result;
}

StringTree generateEnum(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringTree& methodDefinitions) {
	auto nodeBrand = capnp::defaultValue<Brand>();
		
	auto node = getNode(request, nodeId);
	
	KJ_REQUIRE(node.isEnum());
	auto asEnum = node.getEnum();
	auto enumerants = asEnum.getEnumerants();
	
	auto name = nodeName(nodeId, request);
	auto fullName = cppNodeTypeName(nodeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), request);
	
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

StringTree generateStruct(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringTree& methodDefinitions) {
	auto nodeBrand = capnp::defaultValue<Brand>();
		
	auto node = getNode(request, nodeId);
	KJ_REQUIRE(node.isStruct());
	auto asStruct = node.getStruct();
	
	auto name = nodeName(nodeId, request);
	auto fullName = cppNodeTypeName(nodeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), request);
	
	StringTree result = strTree(
		generateTemplateHeader(request, nodeId),
		// "struct CupnpVal<", cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request), "> {\n",
		"struct ", name.flatten(), "{\n",
		"	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;\n",
		"	\n",
		"	uint64_t structure;\n",
		"	cupnp::Location data;\n",
		"	\n",
		"	inline CUPNP_FUNCTION ", name.flatten(), "(uint64_t structure, cupnp::Location data) :\n",
		"		structure(structure),\n",
		"		data(data)\n",
		"	{\n",
		"		cupnp::validateStructPointer(structure, data);\n",
		"	}\n",
		"	\n",
		"	inline CUPNP_FUNCTION bool isDefault() { return structure == 0; }\n",
		"	\n",
		"	friend CUPNP_FUNCTION void cupnp::swapData<", name.flatten(), ">(", name.flatten(), "&, ", name.flatten(), "&); \n",
		"	\n"
	);
	
	result = strTree(
		mv(result),
		generateNested(request, nodeId, methodDefinitions)
	);
	
	methodDefinitions = strTree(
		mv(methodDefinitions),
		"// ===== struct ", fullName.flatten(), " =====\n",
		"\n"
	);
	
	if(canSpecializeOn(request, nodeId)) {
		auto specializationTemplateHeader = generateAllTemplateHeaders(request, nodeId);
		
		if(specializationTemplateHeader.size() == 0)
			specializationTemplateHeader = strTree("template<>\n");
		
		auto capnpName = cppNodeTypeName(nodeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), request, true);
		
		methodDefinitions = strTree(
			mv(methodDefinitions),
			"// CuFor specializaation\n",
			"namespace cupnp {\n",
			mv(specializationTemplateHeader),
			"struct CuFor_<", capnpName.flatten(), "> { using Type = ", fullName.flatten(), "; }; \n",
			"} // namespace ::cupnp\n",
			"\n"
		);
	}
	
	for(auto field : asStruct.getFields()) {
		switch(field.which()) {
			case Field::GROUP: {
				auto typeId = field.getGroup().getTypeId();
				auto typeName = cppNodeTypeName(typeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), request);
				
				if(field.getDiscriminantValue() != 0xffff) {
					KJ_REQUIRE(asStruct.getDiscriminantCount() > 0);
					
					result = strTree(
						mv(result),
						generateMethod(
							request, nodeId, methodDefinitions,
							strTree("bool"), strTree("is", camelCase(field.getName(), true), "() const"),
							strTree(
							"	return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data) == ", field.getDiscriminantValue(), ";\n"
							)
						),
						"	\n"
					);
				
					result = strTree(
						mv(result),
						generateMethod(
							request, nodeId, methodDefinitions,
							typeName.flatten(), strTree("mutate", camelCase(field.getName(), true), "()"),
							strTree(
							"	cupnp::setDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data, ", field.getDiscriminantValue(), ");\n",
							"	return ", typeName.flatten(), "(structure, data);\n"
							)
						)
					);
				} else {
					result = strTree(
						mv(result),
						generateMethod(
							request, nodeId, methodDefinitions,
							typeName.flatten(), strTree("mutate", camelCase(field.getName(), true), "()"),
							strTree(
							"	return ", typeName.flatten(), "(structure, data);\n"
							)
						)
					);
				}
				
				result = strTree(
					mv(result),
					generateMethod(
						request, nodeId, methodDefinitions,
						strTree("const ", typeName.flatten()), strTree("get", camelCase(field.getName(), true), "() const"),
						strTree(
						"	return ", typeName.flatten(), "(structure, data);\n"
						)
					),
					"\n"
				);
				break;
			}
			case Field::SLOT: {
				auto slot = field.getSlot();
				
				auto type = slot.getType();
				auto typeName = cppTypeName(type, nodeId, capnp::defaultValue<Brand>(), request);
				auto fieldType = type.which() == Type::ENUM ? str("uint16_t") : typeName.flatten();
				
				auto subName = camelCase(field.getName(), true);
				auto enumName = enumCase(field.getName());
				
				auto pointerField = [&]() {
				};
						
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
									request, nodeId, methodDefinitions,
									typeName.flatten(), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"	if(cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data) != ", field.getDiscriminantValue(), ")\n",
										"		return ", castIfEnum(cppDefaultValue(slot.getDefaultValue()), typeName.flatten()), ";\n",
										"	\n",
										"	return ", castIfEnum(strTree("cupnp::get", accessorSuffix, "(structure, data, ", cppDefaultValue(slot.getDefaultValue()), ")"), typeName.flatten()), ";\n"
									)
								),
								generateMethod(
									request, nodeId, methodDefinitions,
									strTree("void"), strTree("set", subName.asPtr(), "(", typeName.flatten(), " newVal)"),
									strTree(
										"	cupnp::set", accessorSuffix, "(structure, data, ", cppDefaultValue(slot.getDefaultValue()), ", ", castIfEnum(strTree("newVal"), str(fieldType)), ");\n",
										"	cupnp::setDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data, ", field.getDiscriminantValue(), ");\n"
									)
								),
								"\n"
							);
						} else {
							// Getters and setters for non-unionized primitive fields
							result = strTree(
								mv(result),
								generateMethod(
									request, nodeId, methodDefinitions,
									typeName.flatten(), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"	return ", castIfEnum(strTree("cupnp::get", accessorSuffix, "(structure, data, ", cppDefaultValue(slot.getDefaultValue()), ")"), typeName.flatten()), ";\n"
									)
								),
								generateMethod(
									request, nodeId, methodDefinitions,
									strTree("void"), strTree("set", subName.asPtr(), "(", typeName.flatten(), " newVal)"),
									strTree(
										"	cupnp::set", accessorSuffix, "(structure, data, ", cppDefaultValue(slot.getDefaultValue()), ", ", castIfEnum(strTree("newVal"), str(fieldType)), ");\n"
									)
								),
								"\n"
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
									request, nodeId, methodDefinitions,
									strTree("bool"), strTree("has", subName.asPtr(), "() const"),
									strTree(
										"	return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data) == ", field.getDiscriminantValue(), ";\n"
									)
								)
							);
						}
						
						break;
					
					case Type::TEXT:
					case Type::DATA:
					case Type::LIST:
					case Type::STRUCT:
					case Type::INTERFACE:
					case Type::ANY_POINTER: {
						// Only allow default values in message if they are non-standard
						bool nonstandardDefault = slot.hasDefaultValue() && (slot.getDefaultValue().hasStruct() || slot.getDefaultValue().hasList() || slot.getDefaultValue().hasAnyPointer());
						
						result = strTree(
							mv(result),
							"	inline static const unsigned char ", enumName.asPtr(), "_DEFAULT_VALUE [] = ", generateValueAsBytes(slot.getDefaultValue()), ";\n",
							
							generateMethod(
								request, nodeId, methodDefinitions,
								typeName.flatten(), strTree("mutate", subName.asPtr(), "()"),
								strTree(
									"	CUPNP_REQUIRE(nonDefault", subName.asPtr(), "());\n",
									"	return cupnp::mutatePointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(structure, data);\n"
								)
							),
							"\n"
						);
						
						
						if(field.getDiscriminantValue() != 0xffff) {
							KJ_REQUIRE(asStruct.getDiscriminantCount() > 0);
							
							result = strTree(
								mv(result),
							
								generateMethod(
									request, nodeId, methodDefinitions,
									strTree("const ", typeName.flatten()), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"	if(cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data) != ", field.getDiscriminantValue(), ")\n",
										"		return cupnp::getPointer<", typeName.flatten(), ">(reinterpret_cast<const capnp::word*>(", enumName.asPtr(), "_DEFAULT_VALUE));\n",
										"	\n",
										"	return cupnp::getPointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(structure, data, reinterpret_cast<const capnp::word*>(", enumName.asPtr(), "_DEFAULT_VALUE));\n"
									)
								),
								generateMethod(
									request, nodeId, methodDefinitions,
									strTree("bool"), strTree("nonDefault", subName.asPtr(), "() const"),
									strTree(
										"	return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data) == ", field.getDiscriminantValue(), " && cupnp::hasPointerField<", slot.getOffset(), ">(structure, data);\n"
									)
								),
								generateMethod(
									request, nodeId, methodDefinitions,
									strTree("bool"), strTree("has", subName.asPtr(), "() const"),
									strTree(
										"	return cupnp::getDiscriminant<", asStruct.getDiscriminantOffset(), ">(structure, data) == ", field.getDiscriminantValue(), ";\n"
									)
								),
								"\n"
							);
						} else {
							result = strTree(
								mv(result),
								
								generateMethod(
									request, nodeId, methodDefinitions,
									strTree("const ", typeName.flatten()), strTree("get", subName.asPtr(), "() const"),
									strTree(
										"	return cupnp::getPointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(structure, data, reinterpret_cast<const capnp::word*>(", enumName.asPtr(), "_DEFAULT_VALUE));\n"
									)
								),
								generateMethod(
									request, nodeId, methodDefinitions,
									strTree("bool"), strTree("nonDefault", subName.asPtr(), "() const"),
									strTree(
										"	return cupnp::hasPointerField<", slot.getOffset(), ">(structure, data);\n"
									)
								),
								"\n"
							);
						}
						break;
					}
				}
				break;
			}
		}
	}
	
	result = strTree(
		mv(result), 	
		"}; // struct ", fullName.flatten(), "\n\n"
	);
	
	return mv(result);
}

StringTree generateNode(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringTree& methodDefinitions) {
	StringTree result;
	
	auto node = getNode(request, nodeId);
	
	if(node.isStruct()) {		
		result = strTree(mv(result), generateStruct(request, node.getId(), methodDefinitions));
	}
	
	if(node.isInterface()){
		result = strTree(mv(result), generateInterface(request, node.getId(), methodDefinitions));
	}
	
	if(node.isEnum()) {
		result = strTree(mv(result), generateEnum(request, node.getId(), methodDefinitions));
	}
	
	return mv(result);
}

StringTree generateNested(CodeGeneratorRequest::Reader request, uint64_t nodeId, StringTree& methodDefinitions) {
	StringTree result;
	
	auto node = getNode(request, nodeId);
	
	if(node.isStruct()) {
		auto structNode = node.getStruct();
		
		for(auto field : structNode.getFields()) {
			if(!field.isGroup())
				continue;
			
			auto groupField = field.getGroup();
			result = strTree(mv(result), generateNode(request, groupField.getTypeId(), methodDefinitions));
		}
	}
	
	for(auto child : node.getNestedNodes()) {
		result = strTree(mv(result), generateNode(request, child.getId(), methodDefinitions));
	}
	
	return mv(result);
}

void generateRequested(CodeGeneratorRequest::Reader request) {
	auto fs = kj::newDiskFilesystem();
	auto& cwd = fs->getCurrent();
	
	for(auto fileNode : request.getRequestedFiles()) {
		/*StringTree result = generateNode(request, fileNode.getId());
		
		result = strTree(
			"#pragma once \n",
			"\n",
			"#include <cupnp/cupnp.h>\n",
			"\n",
			"namespace cupnp {\n",
			"\n",
			mv(result),
			"\n",
			"}\n // namespace cupnp"			
		);*/
		
		StringTree namespaceName;
		
		//if(rootNode.isFile()) {
		auto node = getNode(request, fileNode.getId());
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
		
		StringTree forwardDeclarations = generateForwardDeclarations(namespaceName.flatten(), request, fileNode.getId());
		StringTree kindOverrides = generateKindOverrides(namespaceName.flatten(), request, fileNode.getId());
		
		StringTree methodDefinitions;
		StringTree declarations = generateNested(request, fileNode.getId(), methodDefinitions);
		
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
		outFile -> writeAll(str(result));
	}
	
	// return mv(result);
}

void mainFunc(kj::StringPtr programName, kj::ArrayPtr<const kj::StringPtr> args) {
	//KJ_LOG(WARNING, "Initiating compilation");
    capnp::ReaderOptions options;
    options.traversalLimitInWords = 1 << 30;	
	
	capnp::StreamFdMessageReader input(0, options);
	auto root = input.getRoot<CodeGeneratorRequest>();
	
	// Format input
	/*KJ_LOG(WARNING, "Formatting input");
	capnp::TextCodec outputCodec;
	outputCodec.setPrettyPrint(true);
	kj::String result = outputCodec.encode(root);
	KJ_LOG(WARNING, result);*/
	
	generateRequested(root);
	
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
}