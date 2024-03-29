#include "fscpy.h"
#include "async.h"
#include "assign.h"

#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/schema-loader.h>
#include <capnp/any.h>
#include <capnp/generated-header-support.h>

#include <kj/string-tree.h>

#include <fsc/data.h>
#include <fsc/services.h>

#include <cstdint>
#include <cctype>

#include <set>

using capnp::RemotePromise;
using capnp::Response;
using capnp::DynamicCapability;
using capnp::DynamicStruct;
using capnp::DynamicValue;
using capnp::StructSchema;
using capnp::InterfaceSchema;
using capnp::AnyPointer;

using namespace fscpy;

using kj::str;

	
kj::String fscpy::memberName(kj::StringPtr name) {
	auto newName = kj::str(name);
	
	static const std::set<kj::StringPtr> reserved({
		// Python keywords
		"False", "None", "True", "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",
		"finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
		"return", "try", "while", "witdh", "yield", "async",
	});
	
	if(newName.endsWith("_") || (newName.startsWith("init") && newName != "init") || reserved.count(newName) > 0)
		newName = kj::str(newName, "_");
	
	return newName;
}

namespace {

/**
 * \internal
 * Handle for a py::object whose name shows up as "PromiseForResult" in signatures
 */
struct PromiseHandle {
	py::object pyPromise;
};



}

namespace pybind11 { namespace detail {
	template<>
	struct type_caster<PromiseHandle> {		
		PYBIND11_TYPE_CASTER(PromiseHandle, const_name("PromiseForResult"));
		
		bool load(handle src, bool convert) {
			return false;		
		}
		
		static handle cast(PromiseHandle src, return_value_policy policy, handle parent) {
			return src.pyPromise.inc_ref();
		}
	};
}}

namespace {

enum class FSCPyClassType {
	BUILDER, READER, PIPELINE
};

kj::String qualName(py::object scope, kj::StringPtr name) {
	if(py::hasattr(scope, "__qualname__"))
		return kj::str(scope.attr("__qualname__").cast<kj::StringPtr>(), ".", name);
	else
		return kj::heapString(name);
}

kj::String sanitizedStructName(kj::StringPtr input) {
	KJ_IF_MAYBE(pLoc, input.findFirst('$')) {
		// Method-local structs have a method$What name, this needs to be renamed
		auto head = input.slice(0, *pLoc);
		auto tail = input.slice(*pLoc + 1);
		
		// return str(tail, "For_", head);
		return str(tail);
	}
	
	return str(input);
}

// Declaration for recursive calls
py::object interpretSchema(fscpy::Loader& loader, uint64_t id, py::object rootScope, Maybe<py::object> methodScope = nullptr);

py::object interpretStructSchema(fscpy::Loader& loader, capnp::StructSchema schema, py::object rootScope, py::object scope) {	
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	auto structName = sanitizedStructName(schema.getUnqualifiedName());
	
	py::module_ collections = py::module_::import("collections");
	py::type mappingAbstractBaseClass = collections.attr("abc").attr("Mapping");
	
	auto maybeSource = loader.sourceInfo.find(schema.getProto().getId());
	
	py::dict attrs;
	attrs["__qualname__"] = qualName(scope, structName);
	attrs["__module__"] = moduleName;
	
	attrs["__init__"] = py::cpp_function([]() {
		KJ_UNIMPLEMENTED("Do not directly create instances of this class. Use StructType.newMessage() instead to create Builder subclasses");
	});
		
	KJ_IF_MAYBE(pSrc, maybeSource) {
		auto comment = pSrc -> getDocComment();
		if(comment.size() > 0) {
			auto docString = kj::str("For more information about this class, see the help documentation of ", structName, ".Reader or ", structName, ".Builder");
			attrs["__doc__"] = py::str(docString.cStr());
		}
	} else {
		// KJ_DBG("Class has no doc info", structName);
	}
	
	py::object output = (*baseMetaType)(structName, py::make_tuple(), attrs);
	
	// Prevent recursion
	(*globalClasses)[py::cast(schema.getProto().getId())] = output;
	
	py::dict instanceAttrs;
	instanceAttrs["__qualname__"] = qualName(output, "Instance");
	instanceAttrs["__module__"] = moduleName;
	py::object instanceCls = (*baseMetaType)("Instance", py::make_tuple(), instanceAttrs);
	
	output.attr("Instance") = instanceCls;
		
	// Create Builder, Reader, and Pipeline classes
	for(int i = 0; i < 3; ++i) {
		FSCPyClassType classType = (FSCPyClassType) i;
		
		py::dict attributes;
		for(StructSchema::Field field : schema.getFields()) {
			kj::StringPtr rawName = field.getProto().getName();
			kj::String name = memberName(rawName);
			
			using Field = capnp::schema::Field;
			
			auto type = field.getType();
			
			// Only emit pipeline fields for struct and interface fields
			if(classType == FSCPyClassType::PIPELINE && !type.isStruct() && !type.isInterface())
				break;
			
			if(classType == FSCPyClassType::BUILDER) {
				kj::String nameUpper = kj::heapString(rawName);
				nameUpper[0] = toupper(name[0]);
				
				if(type.isList() || type.isData() || type.isText()) {
					auto initializerName = str("init", nameUpper);
					KJ_IF_MAYBE(pFound, schema.findFieldByName(initializerName)) {
					} else {
						attributes[initializerName.cStr()] =  methodDescriptor(py::cpp_function(
							[field](DynamicStructBuilder& builder, size_t n) {
								return builder.initList(field.getProto().getName(), n);
							}
						));
					}
				}
			}
			
			kj::String docString;
			KJ_IF_MAYBE(pSrc, maybeSource) {
				auto comment = pSrc -> getMembers()[field.getIndex()].getDocComment().asReader();
				// KJ_DBG("Field comment", structName, field.getProto().getName(), field.getIndex(), comment);
				if(comment.size() > 0) {
					docString = kj::heapString(comment);
				}
			}
			
			attributes[name.cStr()] = FieldDescriptor(field, mv(docString));
		}
		
		for(StructSchema::Field field : schema.getUnionFields()) {
			kj::StringPtr rawName = field.getProto().getName();
			kj::String name = memberName(rawName);
			
			using Field = capnp::schema::Field;
			
			auto type = field.getType();
			if(classType == FSCPyClassType::BUILDER) {
			
				kj::String nameUpper = kj::heapString(rawName);
				nameUpper[0] = toupper(name[0]);
				
				if(type.isStruct()) {
					auto initializerName = str("init", nameUpper);
					KJ_IF_MAYBE(pFound, schema.findFieldByName(initializerName)) {
					} else {
						attributes[initializerName.cStr()] =  methodDescriptor(py::cpp_function(
							[field](DynamicStructBuilder& builder) {
								return builder.init(field.getProto().getName());
							}
						));
					}
				}
			}
		}
		
		// Determine base class and class suffix
		py::type baseClass = py::type::of(py::none());
		kj::StringPtr suffix;
		kj::StringTree docString;
		
		switch(classType) {
			case FSCPyClassType::BUILDER: 
				baseClass = py::type::of<DynamicStructBuilder>();
				suffix = "Builder";
				docString = strTree("A modifidable instance of ", qualName(scope, structName));
				break;
				
			case FSCPyClassType::READER :
				baseClass = py::type::of<DynamicStructReader>();
				suffix = "Reader";
				docString = strTree("A read-only instance of ", qualName(scope, structName));
				break;
				
			case FSCPyClassType::PIPELINE:
				baseClass = py::type::of<DynamicStructPipeline>();
				suffix = "Pipeline";
				docString = strTree("A pipeline for ", qualName(scope, structName), " that"
					"allows access to capabilities returned in the future by an on-going call"
				);
				break;  
		}
		
		KJ_IF_MAYBE(pSrc, maybeSource) {
			auto comment = pSrc -> getDocComment();
			if(comment.size() > 0) {
				docString = kj::strTree(
					mv(docString),
					"\n\n --- Documentation for class ", qualName(scope, structName), " ---\n\n",
					comment
				);
			}
		}
		
		attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
			[baseClass, classType](py::object self, py::args args, py::kwargs kwargs) {
				baseClass.attr("__init__")(self, *args, **kwargs);
			}
		));
		
		// Disable dynamic attributes
		// attributes["__slots__"] = py::tuple();
		
		// Determine metaclass and build a new type with the given suffix
		// py::type metaClass = py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject*>(&PyType_Type));//py::type::of(baseClass);
		py::type metaClass = py::type::of(baseClass);
		
		attributes["__qualname__"] = qualName(output, suffix);
		attributes["__module__"] = moduleName;
		
		attributes["__slots__"] = py::make_tuple();
		attributes["__doc__"] = py::str(docString.flatten().cStr());
		
		py::tuple baseClassTuple =
			classType == FSCPyClassType::PIPELINE ?
				py::make_tuple(baseClass)
			:
				py::make_tuple(instanceCls, baseClass)
			;
				
			
		py::object newCls = (*baseMetaType)(kj::str(suffix).cStr(), baseClassTuple, attributes);
				
		output.attr(suffix.cStr()) = newCls;
	}
	
	// Create Promise class
	

	// Note: The mixing of Promise and Pipeline had to be removed because of conflicts between the generated and the asyncio.Future interface.
	/*
	{
		py::dict attributes;
		
		py::type promiseBase = futureType();
		py::type pipelineBase = output.attr("Pipeline");
		
		attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
			[promiseBase, pipelineBase](py::object self, py::object future, py::object pipeline) {
				promiseBase.attr("__init__")(self, future);
				pipelineBase.attr("__init__")(self, pipeline);
			}
		));
		
		kj::String suffix = kj::str("Promise");
		attributes["__qualname__"] = qualName(output, suffix);
		attributes["__module__"] = moduleName;
		
		//py::type metaClass = py::type::of(promiseBase);
		py::type metaClass = py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject*>(&PyType_Type));
		
		// py::object newCls = (*baseMetaType)(kj::str(structName, ".", suffix).cStr(), py::make_tuple(promiseBase, pipelineBase), attributes);
		output.attr(suffix.cStr()) = newCls;
	}
	*/
		
	output.attr("newMessage") = py::cpp_function(
		[schema](py::object copyFrom, size_t initialSize) mutable {
			auto msg = kj::heap<capnp::MallocMessageBuilder>(initialSize);
			
			capnp::DynamicStruct::Builder builder;
			if(copyFrom.is_none()) {
				builder = msg->initRoot<capnp::DynamicStruct>(schema);
			} else {
				// Let's try using our assignment logic
				assign(*msg, schema, copyFrom);
				builder = msg->getRoot<capnp::DynamicStruct>(schema);
			}
			
			return DynamicStructBuilder(mv(msg), builder);
		},
		py::name("newMessage"),
		py::scope(output),
		py::arg("copyFrom") = py::none(),
		py::arg("initialSize") = 1024
	);
	
	/*output.attr("castAs") = py::cpp_function(
		[schema](py::object input) -> py::object {
			py::detail::make_caster<capnp::DynamicStruct::Reader> readerCaster;
			py::detail::make_caster<capnp::DynamicStruct::Builder> builderCaster;
			
			py::object result;
			
			if(builderCaster.load(input, false)) {
				capnp::AnyStruct::Builder asBuilder = builderCaster.operator capnp::DynamicStruct::Builder&();
				result = py::cast(asBuilder.as<capnp::DynamicStruct>(schema));
			} else if(readerCaster.load(input, false)) {
				capnp::AnyStruct::Reader asReader = readerCaster.operator capnp::DynamicStruct::Reader&();
				result = py::cast(asReader.as<capnp::DynamicStruct>(schema));
			} else {
				KJ_FAIL_REQUIRE("Object is not a struct reader or builder");
			}
			
			result.attr("_castFrom") = input;
			return result;
		}
	);
		
	output.attr("_initRootAs") = py::cpp_function(
		[schema](py::object src) mutable {
			auto& msg = py::cast<capnp::MessageBuilder&>(src);
			
			// We use DynamicValue instead of DynamicStruct to engage our type-dependent dispatch
			capnp::DynamicValue::Builder builder = msg.initRoot<capnp::DynamicStruct>(schema);			
			py::object result = py::cast(builder);
			
			result.attr("_msg") = src;
			
			return result;
		},
		py::name("newMessage"),
		py::scope(output),
		py::arg("messageBuilder")
	);*/
	
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr rawName = field.getProto().getName();
		kj::String name = memberName(rawName);
		
		using Field = capnp::schema::Field;
		
		switch(field.getProto().which()) {
			case Field::GROUP: {
				capnp::StructSchema subType = field.getType().asStruct();
				output.attr(name.cStr()) = interpretSchema(loader, subType.getProto().getId(), rootScope);
				break;
			}
			
			case Field::SLOT: {} // Already processed above
		}
	}
	
	return output;
}

py::object interpretEnumSchema(capnp::EnumSchema schema, py::object scope) {	
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	
	auto enumName = schema.getUnqualifiedName();
	
	py::dict attrs;
	attrs["__qualname__"] = qualName(scope, enumName);
	attrs["__module__"] = moduleName;
	
	attrs["__init__"] = py::cpp_function([]() {
		KJ_UNIMPLEMENTED("Do not create instances of this class. Use EnumType.get() instead");
	});
	
	py::object output = (*baseMetaType)(enumName, py::make_tuple(), attrs);
	
	output.attr("get") = py::cpp_function(
		[schema](DynamicValueReader from) -> EnumInterface {
			if(from.getType() == capnp::DynamicValue::ENUM) {
				auto enumInput = from.as<capnp::DynamicEnum>();
				KJ_REQUIRE(enumInput.getSchema().getProto().getId() == schema.getProto().getId(), "Incorrect enum type");
				return capnp::DynamicEnum(schema, enumInput.getRaw());
			}
			
			if(from.getType() == capnp::DynamicValue::TEXT) {
				return capnp::DynamicEnum(schema.getEnumerantByName(from.as<capnp::Text>()));
			}
			
			if(from.getType() == capnp::DynamicValue::UINT || from.getType() == capnp::DynamicValue::INT) {
				return capnp::DynamicEnum(schema, from.as<uint16_t>());
			}
			
			// KJ_DBG(from, from.getType());
			throw std::invalid_argument("Input must be an enum value, a string, or an integer");
		}
	);
	
	// Values list
	py::list vals;
	for(auto enumerant : schema.getEnumerants()) {
		vals.append(py::cast(EnumInterface(enumerant)));
	}
	output.attr("values") = vals;
	
	return output;
}

py::object interpretInterfaceSchema(fscpy::Loader& loader, capnp::InterfaceSchema schema, py::object rootScope, py::object scope) {		
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	auto methods = schema.getMethods();
	
	auto collections = py::module_::import("collections");
	auto orderedDict = collections.attr("OrderedDict");
	
	// We need dynamic resolution to get our base capability client
	// Static resolution fails as we have overridden the type caster
	/*py::object baseObject = py::cast(DynamicCapability::Client());
	py::object baseClass = py::type::of(baseObject);*/
	py::object baseClass = py::type::of<DynamicCapabilityClient>();
	
	py::type metaClass = py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject*>(&PyType_Type));
	//py::type metaClass = py::type::of(baseClass);
	
	py::dict outerAttrs;
	py::dict clientAttrs;
	
	outerAttrs["schema"] = py::cast(schema);
	
	for(auto i : kj::indices(methods)) {
		auto pyMethod = loader.makeInterfaceMethod(methods[i]);
		clientAttrs[pyMethod.attr("__name__")] = pyMethod;
	}
		
	py::object methodHolder = simpleObject();
	methodHolder.attr("__name__") = "methods";
	methodHolder.attr("__qualname__") = qualName(scope, str(schema.getUnqualifiedName(), ".methods"));
	methodHolder.attr("__module__") = moduleName;
	outerAttrs["methods"] = methodHolder;
	
	outerAttrs["newDeferred"] = py::cpp_function(
		[schema](Promise<capnp::DynamicCapability::Client> promise) {			
			capnp::Capability::Client untyped = mv(promise);
			return untyped.castAs<capnp::DynamicCapability>(schema);
		}
	);
	
	outerAttrs["newDisconnected"] = py::cpp_function(
		[schema](kj::StringPtr disconnectReason) {
			capnp::Capability::Client untyped(capnp::newBrokenCap(disconnectReason));
			return untyped.castAs<capnp::DynamicCapability>(schema);
		},
		py::arg("disconnectReason") = "disconnected"
	);
	
	outerAttrs["castAs"] = py::cpp_function(
		[schema](capnp::DynamicCapability::Client input) {
			capnp::Capability::Client asGeneric(mv(input));
			return asGeneric.castAs<capnp::DynamicCapability>(schema);
		}
	);
	
	auto outerName = qualName(scope, schema.getUnqualifiedName());
	outerAttrs["__qualname__"] = outerName;
	outerAttrs["__module__"] = moduleName;
	
	py::object outerCls = metaClass(schema.getUnqualifiedName(), py::make_tuple(), outerAttrs);
	
	auto innerName = qualName(outerCls, "Client");
	clientAttrs["__qualname__"] = innerName;
	clientAttrs["__module__"] = moduleName;
	
	py::list bases;
	for(auto baseType : schema.getSuperclasses()) {
		auto id = baseType.getProto().getId();
		
		if(globalClasses -> contains(id))
			bases.append((*globalClasses)[py::cast(id)].attr("Client"));
		else
			py::print("Missing base class when creating class", innerName, ", id =", id);
	}
	
	if(bases.size() == 0)
		bases.append(baseClass);
	
	py::object clientCls = (*baseMetaType)("Client", py::eval("tuple")(bases), clientAttrs);
	outerCls.attr("Client") = clientCls;
	
	// Define server class
	py::dict serverAttrs;
	py::object serverBaseClass = py::type::of<DynamicCapabilityServer>();
	
	serverAttrs["__init__"] = methodDescriptor(py::cpp_function(
		[baseInit = serverBaseClass.attr("__init__"), schema](py::object pySelf) {
			baseInit(pySelf, py::cast(schema));
		}
	));
	serverAttrs["__qualname__"] = qualName(outerCls, "Server");
	serverAttrs["__module__"] = moduleName;
	
	py::object serverCls = (*baseMetaType)("Server", py::make_tuple(serverBaseClass), serverAttrs);
	outerCls.attr("Server") = serverCls;
	
	// Remember the class before interpreting children to make sure we don't recurse
	(*globalClasses)[py::cast(schema.getProto().getId())] = outerCls;
	
	for(auto i : kj::indices(methods)) {
		auto method = methods[i];
		kj::String name = memberName(method.getProto().getName());
		
		py::object holder = simpleObject();
		holder.attr("__qualname__") = qualName(scope, str(schema.getUnqualifiedName(), ".methods.", name));
		holder.attr("__name__") = name;
		holder.attr("__module__") = moduleName;
		holder.attr("desc") = str("Auxiliary classes (Params and/or Results) for method ", name);
		
		py::object pyParamType = interpretSchema(loader, method.getParamType().getProto().getId(), rootScope, holder);
		py::object pyResultType = interpretSchema(loader, method.getResultType().getProto().getId(), rootScope, holder);
		
		KJ_REQUIRE(!pyParamType.is_none());
		KJ_REQUIRE(!pyResultType.is_none());
		
		holder.attr(pyParamType.attr("__name__")) = pyParamType;
		holder.attr(pyResultType.attr("__name__")) = pyResultType;
		
		methodHolder.attr(name.cStr()) = holder;
	}
	
	return outerCls;
}

py::object interpretConstSchema(capnp::ConstSchema schema, py::object scope) {	
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	
	py::object result = py::cast(new ConstantValue(schema));
	
	//auto outerName = qualName(scope, schema.getUnqualifiedName());
	//result.attr("__qualname__") = outerName;
	//result.attr("__module__") = moduleName;
	
	return result;
}

py::object interpretSchema(fscpy::Loader& loader, uint64_t id, py::object rootScope, Maybe<py::object> methodScope) {	
	if(globalClasses->contains(id))
		return (*globalClasses)[py::cast(id)];
	
	KJ_IF_MAYBE(dontCare, loader.capnpLoader.tryGet(id)) {
	} else {
		return py::none();
	}
	
	capnp::Schema schema = loader.capnpLoader.get(id);
		
	// Find parent object
	py::object parent;
	{
		uint64_t scopeId = schema.getProto().getScopeId();
		if(scopeId != 0) {
			parent = interpretSchema(loader, scopeId, rootScope);
						
			// Interpreting the parent schema can cause the node to
			// be generated. Re-check the dict.
			if(globalClasses->contains(id))
				return (*globalClasses)[py::cast(id)];
		} else {
			KJ_IF_MAYBE(pMethodScope, methodScope) {
				parent = *pMethodScope;
			} else {
				parent = rootScope;
			}
		}
	}
	
	if(parent.is_none())
		parent = rootScope;
	
	py::str moduleName = py::hasattr(parent, "__module__") ? parent.attr("__module__") : parent.attr("__name__");
	py::object output = py::none();
	
	switch(schema.getProto().which()) {
		case capnp::schema::Node::STRUCT:
			output = interpretStructSchema(loader, schema.asStruct(), rootScope, parent);
			break;
		case capnp::schema::Node::INTERFACE:
			output = interpretInterfaceSchema(loader, schema.asInterface(), rootScope, parent);
			break;
		case capnp::schema::Node::FILE:
			output = rootScope;
			break;
		case capnp::schema::Node::ENUM:
			output = interpretEnumSchema(schema.asEnum(), parent);
			break;
		case capnp::schema::Node::CONST:
			output = interpretConstSchema(schema.asConst(), parent);
			break;
		
		default:
			py::dict attrs;
			attrs["__qualname__"] = qualName(parent, schema.getUnqualifiedName()).asPtr();
			attrs["__module__"] = moduleName;
			
			output = (*baseMetaType)(schema.getUnqualifiedName().cStr(), py::make_tuple(), attrs);
			break;
	}
	
	KJ_REQUIRE(!output.is_none(), "Failed to interpret node, schema is incomplete", id);
	
	// Remember the class before interpreting children to make sure we don't recurse
	(*globalClasses)[py::cast(id)] = output;
		
	// Interpret child objects
	for(auto nestedNode : schema.getProto().getNestedNodes()) {
		py::object subObject = interpretSchema(loader, nestedNode.getId(), rootScope);
		
		if(subObject.is_none())
			continue;
			
		output.attr(nestedNode.getName().cStr()) = subObject;
	}
		
	return output;
}

}

// ================== Implementation of typeName ==========================

kj::StringTree fscpy::typeName(capnp::Type type) {
	using ST = capnp::schema::Type;
	using kj::strTree;
	
	switch(type.which()) {
		case ST::VOID:
			return strTree("Any");
			
		case ST::BOOL:
			return strTree("bool");
			
		case ST::INT8:
		case ST::INT16:
		case ST::INT32:
		case ST::INT64:
		case ST::UINT8:
		case ST::UINT16:
		case ST::UINT32:
		case ST::UINT64:
			return strTree("int");
		
		case ST::FLOAT32:
		case ST::FLOAT64:
			return strTree("real");
		
		case ST::TEXT:
			return strTree("str");
		
		case ST::DATA:
			return strTree("bytes");
		
		case ST::LIST: {
			auto asList = type.asList();
			return strTree("List(", typeName(asList.getElementType()), ")");
		}
		
		case ST::ENUM: {
			auto asEnum = type.asEnum();
			return strTree(asEnum.getUnqualifiedName());
		}
		
		// TODO: Add brand bindings?
		
		case ST::STRUCT: {
			auto asStruct = type.asStruct();
			auto nameStr = asStruct.getUnqualifiedName();
			
			return strTree(sanitizedStructName(nameStr));
		}
		
		case ST::INTERFACE: {
			auto asIntf = type.asInterface();
			return strTree(asIntf.getUnqualifiedName());
		}
		
		case ST::ANY_POINTER:
			return strTree("Any");
		
		default:
			KJ_FAIL_REQUIRE("Unknown type kind");
	}
}

// ================== Implementation of fscpy::Loader =====================

kj::Tuple<kj::StringPtr, kj::StringTree> fscpy::Loader::qualName(capnp::Type type) {
	using Which = capnp::schema::Type::Which;
	
	switch(type.which()) {
		#define HANDLE_TYPE(W, T) \
			case Which::W: return kj::tuple("fusionsc.capnp", kj::strTree(#T));
		
		HANDLE_TYPE(FLOAT32, Float32)
		HANDLE_TYPE(FLOAT64, Float64)
		HANDLE_TYPE(INT8, Int8)
		HANDLE_TYPE(INT16, Int16)
		HANDLE_TYPE(INT32, Int32)
		HANDLE_TYPE(INT64, Int64)
		HANDLE_TYPE(UINT8, UInt8)
		HANDLE_TYPE(UINT16, UInt16)
		HANDLE_TYPE(UINT32, UInt32)
		HANDLE_TYPE(UINT64, UInt64)
		HANDLE_TYPE(BOOL, Bool)
		HANDLE_TYPE(VOID, Void)
		HANDLE_TYPE(TEXT, Text)
		HANDLE_TYPE(DATA, Data)
		
		#undef HANDLE_TYPE
		
		#define HANDLE_TYPE(W, f) \
			case Which::W: return qualName(type.f());
		
		HANDLE_TYPE(INTERFACE, asInterface)
		HANDLE_TYPE(STRUCT, asStruct)
		HANDLE_TYPE(ENUM, asEnum)
		
		#undef HANDLE_TYPE
		
		case Which::LIST: {
			capnp::ListSchema ls = type.asList();
			
			auto addElementType = [&](kj::StringPtr moduleName, kj::StringTree typeName) {
				return kj::tuple("fusionsc.capnp",kj::strTree("List[", moduleName, ".", mv(typeName), "]"));
			};
			
			return kj::apply(addElementType, qualName(ls.getElementType()));
		}
		
		case Which::ANY_POINTER: {
			using Which = capnp::schema::Type::AnyPointer::Unconstrained::Which;
			switch(type.whichAnyPointerKind()) {
				#define HANDLE_TYPE(W, T) \
					case Which::W: return kj::tuple("fusionsc.capnp", kj::strTree(#T));
				
				HANDLE_TYPE(ANY_KIND, AnyPointer)
				HANDLE_TYPE(LIST, AnyList)
				HANDLE_TYPE(STRUCT, AnyStruct)
				HANDLE_TYPE(CAPABILITY, Capability)
				
				#undef HANDLE_TYPE
			}
		}
	}
	
	KJ_FAIL_REQUIRE("Unknown type kind");
}

kj::Tuple<kj::StringPtr, kj::StringTree> fscpy::Loader::qualName(capnp::Schema schema) {
	// Null capability schema
	if(schema.getProto().getId() == capnp::typeId<capnp::Capability>()) {
		return kj::tuple("fusionsc.capnp", kj::strTree("Capability"));
	}
	auto result = kj::strTree(sanitizedStructName(schema.getUnqualifiedName()));
	
	if(schema.isBranded()) {
		auto bindings = schema.getBrandArgumentsAtScope(schema.getProto().getId());
		if(bindings.size() != 0) {
			auto bindingNames = kj::heapArrayBuilder<kj::StringTree>(bindings.size());
			
			auto addBindingName = [&](kj::StringPtr moduleName, kj::StringTree name) {
				bindingNames.add(kj::strTree(moduleName, ".", mv(name)));
			};
			
			for(capnp::Type binding : bindings) {
				kj::apply(addBindingName, qualName(binding));
			}
			
			result = kj::strTree(mv(result), "[", kj::StringTree(bindingNames.finish(), ","), "]");
		}
	} 
	
	// Check if root schema
	KJ_IF_MAYBE(pModuleName, rootModules.find(schema.getProto().getId())) {
		return kj::tuple(pModuleName -> asPtr(), mv(result));
	}
	
	auto scopeId = schema.getProto().getScopeId();
	if(scopeId != 0) {
		auto parent = capnpLoader.get(scopeId, capnp::schema::Brand::Reader(), schema);
		auto parentResult = qualName(parent);
		result = kj::strTree(mv(kj::get<1>(parentResult)), ".", mv(result));
		
		return kj::tuple(kj::get<0>(parentResult), mv(result));
	} else {
		// Check if result type
		KJ_IF_MAYBE(pEntry, resultOfMethod.find(schema.getProto().getId())) {
			auto parentInterface = capnpLoader.get(kj::get<0>(*pEntry), capnp::schema::Brand::Reader(), schema).asInterface();
			auto method = parentInterface.getMethods()[kj::get<1>(*pEntry)];
			auto parentResult = qualName(parentInterface);
			
			return kj::tuple(kj::get<0>(parentResult), kj::strTree(mv(kj::get<1>(parentResult)), ".methods['", method.getProto().getName(), "'].Results"));
		}
		
		// Check if param type
		KJ_IF_MAYBE(pEntry, paramOfMethod.find(schema.getProto().getId())) {
			auto parentInterface = capnpLoader.get(kj::get<0>(*pEntry), capnp::schema::Brand::Reader(), schema).asInterface();
			auto method = parentInterface.getMethods()[kj::get<1>(*pEntry)];
			auto parentResult = qualName(parentInterface);
			
			return kj::tuple(kj::get<0>(parentResult), kj::strTree(mv(kj::get<1>(parentResult)), ".methods['", method.getProto().getName(), "'].Params"));
		}
	}
	
	KJ_FAIL_REQUIRE("No root schema could be found, but node has no parent", schema, schema.getProto().getScopeId());
}

kj::Tuple<kj::StringPtr, kj::StringTree> fscpy::Loader::qualName(capnp::InterfaceSchema::Method method) {
	kj::Tuple<kj::StringPtr, kj::StringTree> interfaceName = qualName(method.getContainingInterface());
	
	return kj::tuple(kj::get<0>(interfaceName), kj::strTree(mv(kj::get<1>(interfaceName)), ".", method.getProto().getName()));
}

kj::String fscpy::Loader::fullName(capnp::Type t) {
	auto qn = qualName(t);
	return kj::str(kj::get<0>(qn), ".", kj::get<1>(qn));
}

bool fscpy::Loader::importNode(uint64_t nodeID, py::module scope) {		
	kj::Function<void(capnp::Schema)> handleSchema = [&](capnp::Schema schema) {
		const uint64_t nodeID = schema.getProto().getId();
		
		if(rootModules.find(nodeID) == nullptr) {
			rootModules.insert(
				nodeID,
				kj::heapString(py::cast<kj::StringPtr>(scope.attr("__name__")))
			);
		}
		
		auto name = memberName(schema.getUnqualifiedName());
		
		switch(schema.getProto().which()) {
			case capnp::schema::Node::STRUCT:
			case capnp::schema::Node::INTERFACE:
			case capnp::schema::Node::ENUM: {
				auto obj = py::cast(schema);
				
				if(py::hasattr(scope, name.cStr()))
					return;
				
				scope.add_object(name.cStr(), obj);
				break;
			}
			case capnp::schema::Node::FILE: {
				// Interpret child objects
				for(auto nestedNode : schema.getProto().getNestedNodes()) {
					auto childSchema = capnpLoader.get(nestedNode.getId());
					handleSchema(childSchema);
				}
				break;
			}
			case capnp::schema::Node::CONST:
				scope.add_object(name.cStr(), py::cast(new ConstantValue(schema.asConst())));
				break;
			default: break;
		}
	};
	
	handleSchema(capnpLoader.get(nodeID));
	
	// Recursively register all methods
	
	kj::Function<void(capnp::Schema)> registerMethods = [&](capnp::Schema schema) {
		switch(schema.getProto().which()) {
			case capnp::schema::Node::INTERFACE: {
				auto methods = schema.getProto().getInterface().getMethods();
				
				for(auto i : kj::indices(methods)) {
					auto paramType = capnpLoader.get(methods[i].getParamStructType());
					if(paramType.getProto().getScopeId() == 0)
						paramOfMethod.insert(methods[i].getParamStructType(), kj::tuple(schema.getProto().getId(), i));
					
					auto resultType = capnpLoader.get(methods[i].getResultStructType());
					if(resultType.getProto().getScopeId() == 0)
						resultOfMethod.insert(methods[i].getResultStructType(), kj::tuple(schema.getProto().getId(), i));
				}
				break;
			}
			
			default: break;
		}
		
		for(auto nestedNode : schema.getProto().getNestedNodes()) {
			auto childSchema = capnpLoader.get(nestedNode.getId());
			registerMethods(childSchema);
		}
	};
	
	registerMethods(capnpLoader.get(nodeID));
	
	return true;
}

bool fscpy::Loader::importNodeIfRoot(uint64_t nodeID, py::module scope) {
	auto schema = capnpLoader.get(nodeID);
	
	// A '$' in the name indicates generated a generated parameter type for methods
	// We attach these to special method objects in the surrounding class
	KJ_IF_MAYBE(dontCare, schema.getUnqualifiedName().findFirst('$')) {
		return false;
	}
	
	uint64_t parentId = schema.getProto().getScopeId();
	
	// If the node has a non-file parent, ignore it
	KJ_IF_MAYBE(pSchema, capnpLoader.tryGet(parentId)) {
		if(!pSchema -> getProto().isFile())
			return false;
	}
	
	return importNode(nodeID, scope);
}

void fscpy::Loader::add(capnp::schema::Node::Reader reader) {
	capnpLoader.loadOnce(reader);
}

void fscpy::Loader::addSource(capnp::schema::Node::SourceInfo::Reader reader) {
	KJ_IF_MAYBE(pSrc, sourceInfo.find(reader.getId())) {
		return;
	}
	
	sourceInfo.insert(reader.getId(), reader);
}

capnp::Schema fscpy::Loader::import(capnp::Schema input) {
	KJ_IF_MAYBE(pSchema, imported.find(input)) {
		return *pSchema;
	}
	
	fsc::Temporary<capnp::schema::Brand> brand;
	extractBrand(input, brand);
	
	capnp::Schema importedSchema = capnpLoader.get(input.getProto().getId(), brand);
	imported.insert(input, importedSchema);
	
	return importedSchema;
}

fscpy::Loader fscpy::defaultLoader;

void fscpy::initLoader(py::module_& m) {
	auto loader = m.def_submodule("loader", "Gives access to the schema loader");
	
	py::dict defaultGlobalScope;
	
	auto ires = py::module::import("importlib_resources");
	auto fscRoot = ires.attr("files")("fusionsc").attr("joinpath")("service");
	
	defaultGlobalScope["fusionsc"] = fscRoot.attr("joinpath")("fusionsc");
	defaultGlobalScope["capnp"] = fscRoot.attr("joinpath")("capnp");
	
	loader.attr("defaultScope") = defaultGlobalScope;
	
	loader.def(
		"getType",
		
		[](uint64_t id) {
			return globalClasses -> attr("get")(id, py::none());
		}
	);
	
	globalClasses = kj::heap<py::dict>();
	
	loader.attr("roots") = py::dict();
	loader.def(
		"parseSchema",
		
		[loader](kj::StringPtr path, py::object scope, py::object root) {			
			parseSchema(root, path, scope, loader.attr("roots"));
		},
		
		py::arg("path"),
		py::arg("destination"),
		py::arg("pathRoot") = py::none()
	);
}
