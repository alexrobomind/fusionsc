#include <capnp/schema.capnp.h>
#include <capnp/serialize.h>
#include <capnp/serialize-text.h>
#include <capnp/any.h>

#include <kj/filesystem.h>

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

kj::StringTree cppTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request);
kj::StringTree cppNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request);

kj::StringTree cppTypeName(Type::Reader type, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request) {
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
		case Type::TEXT: return strTree("capnp::Text");
		case Type::DATA: return strTree("capnp::Data");
		
		case Type::LIST: {
			auto listType = type.getList();
			return strTree("capnp::List<", cppTypeName(listType.getElementType(), scopeId, scopeBrand, request), ">");
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
						case Type::AnyPointer::Unconstrained::ANY_KIND: return strTree("capnp::AnyPointer");
						case Type::AnyPointer::Unconstrained::STRUCT: return strTree("capnp::AnyStruct");
						case Type::AnyPointer::Unconstrained::LIST: return strTree("capnp::AnyList");
						case Type::AnyPointer::Unconstrained::CAPABILITY: return strTree("capnp::Capability");
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
	}
}

static constexpr uint64_t ANNOT_NAMESPACE = 0xb9c6f99ebf805f2cull;
static constexpr uint64_t ANNOT_NAME = 0xf264a779fef191ceull;

kj::StringTree cppNodeTypeName(uint64_t nodeId, Brand::Reader nodeBrand, uint64_t scopeId, Brand::Reader scopeBrand, CodeGeneratorRequest::Reader request) {
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
	
	if(needsTypename)
		result = strTree("typename ", mv(result));
	
	return mv(result);
}

StringTree generateTemplateHeader(CodeGeneratorRequest::Reader request, uint64_t nodeId) {
	kj::Array<Node::Reader> nodes = fullScope(request, nodeId);
	
	kj::Vector<StringTree> paramNames;
	
	for(auto node : nodes) {
		auto nParams = node.getParameters().size();
		
		/*if(nParams == 0)
			continue;
		
		auto paramNames = kj::heapArrayBuilder<StringTree>(nParams);*/
		for(unsigned int i = 0; i < nParams; ++i) {
			paramNames.add(strTree("typename ", parameterName(request, node.getId(), i)));
		}
	}
	
	StringTree result = strTree(
		"template<", StringTree(paramNames.releaseAsArray(), ", "), ">\n"
	);
	
	return mv(result);
}

StringTree generateInterface(CodeGeneratorRequest::Reader request, uint64_t nodeId) {
	auto nodeBrand = capnp::defaultValue<Brand>();
	auto node = getNode(request, nodeId);
	
	KJ_REQUIRE(node.isInterface());
	auto asInterface = node.getInterface();
	
	StringTree result = strTree(
		generateTemplateHeader(request, nodeId),
		"struct CupnpVal<", cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request), "> {\n",
		"	// Interface pointer that holds the capability table offset\n",
		"	uint64_t ptrData;\n",
		"	\n",
		"	CupnpVal(uint64_t structure, cupnp::Location data) : ptrData(structure) {\n",
		"		cupnp::validateInterfacePointer(ptrData);\n",
		"	}\n",
		"	\n",
		"	bool isDefault() { return ref == nullptr; }\n",
		"	\n",
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

StringTree generateStruct(CodeGeneratorRequest::Reader request, uint64_t nodeId) {
	auto nodeBrand = capnp::defaultValue<Brand>();
		
	auto node = getNode(request, nodeId);
	KJ_REQUIRE(node.isStruct());
	auto asStruct = node.getStruct();
	
	StringTree result = strTree(
		generateTemplateHeader(request, nodeId),
		"struct CupnpVal<", cppNodeTypeName(nodeId, nodeBrand, nodeId, nodeBrand, request), "> {\n",
		"	uint64_t structure;\n",
		"	cupnp::Location data;\n",
		"	\n",
		"	inline CupnpVal(uint64_t structure, cupnp::Location data) :\n",
		"		structure(structure),\n",
		"		data(data)\n",
		"	{\n",
		"		cupnp::validateStructPointer(structure, data);\n",
		"	}\n",
		"	\n",
		"	inline bool isDefault() { return structure == 0; }\n",
		"	\n"
	);
	
	for(auto field : asStruct.getFields()) {
		switch(field.which()) {
			case Field::GROUP: {
				auto typeId = field.getGroup().getTypeId();
				auto typeName = cppNodeTypeName(typeId, capnp::defaultValue<Brand>(), nodeId, capnp::defaultValue<Brand>(), request);
				
				result = strTree(
					mv(result),
					"	inline CupnpVal<", typeName.flatten(), "> get", camelCase(field.getName(), true), "() const {\n"
					"		return CupnpVal<", typeName.flatten(), ">(structure, data);\n"
					"	}\n\n"
				);
				break;
			}
			case Field::SLOT: {
				auto slot = field.getSlot();
				
				auto type = slot.getType();
				auto typeName = cppTypeName(type, nodeId, capnp::defaultValue<Brand>(), request);
				
				auto subName = camelCase(field.getName(), true);
				auto enumName = enumCase(field.getName());
				
				auto pointerField = [&]() {
				};
				
				switch(type.which()) {
					case Type::VOID:
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
					case Type::ENUM: {
						result = strTree(
							mv(result),
							"	inline CupnpVal<", typeName.flatten(), "> get", subName.asPtr(), "() const {\n"
							"		return cupnp::getPrimitiveField<", typeName.flatten(), ", ", slot.getOffset(), ", ", cppDefaultValue(slot.getDefaultValue()), ">(structure, data);\n",
							"	}\n\n",
							
							"	inline void set", subName.asPtr(), "(", typeName.flatten(), " newVal) {\n",
							"		cupnp::setPrimitiveField<", typeName.flatten(), ", ", slot.getOffset(), ", ", cppDefaultValue(slot.getDefaultValue()), ">(structure, data, newVal);\n",
							"	}\n\n"
						);
						break;
					}
					
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
							// TODO: These need to be defined in a separate .cu / .cpp file
							"	inline static const unsigned char ", enumName.asPtr(), "_DEFAULT_VALUE = ", generateValueAsBytes(slot.getDefaultValue()), ";\n",
							
							"	inline const CupnpVal<", typeName.flatten(), "> get", subName.asPtr(), "() const {\n",
							"		return cupnp::getPointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(structure, data, ", enumName.asPtr(), "_DEFAULT_VALUE);\n",
							"	}\n\n",
							"	inline CupnpVal<", typeName.flatten(), "> mutate", subName.asPtr(), "() {\n",
							"		return cupnp::mutatePointerField<", typeName.flatten(), ", ", slot.getOffset(), ">(structure, data);\n",
							"	}\n\n",
							"	inline bool has", subName.asPtr(), "() const {\n",
							"		return cupnp::hasPointerField<", slot.getOffset(), ">(structure, data);\n",
							"	}\n\n"
						);
						break;
					}
				}
				break;
			}
		}
	}
	
	result = strTree(
		mv(result), 	
		"};\n\n"
	);
	
	return mv(result);
}

StringTree generateNode(CodeGeneratorRequest::Reader request, uint64_t nodeId) {
	StringTree result;
	
	auto node = getNode(request, nodeId);
	for(auto child : node.getNestedNodes()) {
		auto childNode = getNode(request, child.getId());
		if(childNode.isStruct()) {
			result = strTree(mv(result), generateStruct(request, child.getId()));
		}
		if(childNode.isInterface()) {
			result = strTree(mv(result), generateInterface(request, child.getId()));
		}
		
		result = strTree(mv(result), generateNode(request, child.getId()));
	}
	
	return mv(result);
}

void generateRequested(CodeGeneratorRequest::Reader request) {
	auto fs = kj::newDiskFilesystem();
	auto& cwd = fs->getCurrent();
	
	for(auto fileNode : request.getRequestedFiles()) {
		StringTree result = generateNode(request, fileNode.getId());
		
		auto inputFilename = fileNode.getFilename();
		KJ_LOG(WARNING, inputFilename);
		kj::Path inputFile = kj::Path::parse(inputFilename);
		
		KJ_LOG(WARNING, inputFile);
		
		/*if(baseName.size() < 6 || baseName.slice(baseName.size() - 6) != ".capnp") {
			KJ_LOG(WARNING, "Skipped file because its name does not end with '.capnp'", baseName);
			continue;
		}*/
		
		kj::Vector<char> baseNameVec;
		baseNameVec.addAll(inputFile.basename()[0]);
		baseNameVec.resize(baseNameVec.size() - 6);
		baseNameVec.add('\0');
		
		kj::String baseName(baseNameVec.releaseAsArray());
		KJ_LOG(WARNING, baseName);
		
		kj::String headerName = str(baseName, ".cupnp.h");
		KJ_LOG(WARNING, headerName);
		
		auto outFile = cwd.openFile(kj::Path(headerName), kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
		outFile -> writeAll(str(result));
	}
	
	// return mv(result);
}

int main() {
    capnp::ReaderOptions options;
    options.traversalLimitInWords = 1 << 30;
	
	// Create access to stdin
    capnp::StreamFdMessageReader input(0, options);
	auto root = input.getRoot<CodeGeneratorRequest>();
	
	// Format input
	capnp::TextCodec outputCodec;
	outputCodec.setPrettyPrint(true);
	kj::String result = outputCodec.encode(root);
	
	// Open file
	/*auto fs = kj::newDiskFilesystem();
	auto& cwd = fs->getCurrent();
	
	auto file = cwd.openFile(kj::Path("output.txt"), kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
	
	// Write formatted output
	file->writeAll(result);
	
	auto file2 = cwd.openFile(kj::Path("output2.txt"), kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
	file2 -> writeAll(str(generateRequested(root)));*/
	generateRequested(root);
	
	return 0;
}