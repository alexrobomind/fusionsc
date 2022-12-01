#include "fscpy.h"
#include "async.h"
#include "loader.h"


#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/schema-loader.h>

#include <kj/string-tree.h>

#include <fsc/data.h>
#include <fsc/services.h>

#include <cstdint>
#include <cctype>

using capnp::RemotePromise;
using capnp::Response;
using capnp::DynamicCapability;
using capnp::DynamicStruct;
using capnp::DynamicValue;
using capnp::StructSchema;
using capnp::InterfaceSchema;
using capnp::AnyPointer;

using namespace fscpy;

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
	
kj::String memberName(kj::StringPtr name) {
	auto newName = kj::str(name);
	
	static const std::set<kj::StringPtr> reserved({
		// Python keywords
		"False", "None", "True", "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",
		"finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
		"return", "try", "while", "witdh", "yield", "async",
		
		// Special member names
		"get", "set", "adopt", "disown", "clone", "pretty", "totalSize", "visualize", "items"
	});
	
	if(newName.endsWidth("_") || newName.startsWith("init") || reserved.count(newName) > 0) {
		newName = kj::str(fieldName, "_");
	
	return newName;
}

enum class FSCPyClassType {
	BUILDER, READER, PIPELINE
};

kj::String qualName(py::object scope, kj::StringPtr name) {
	if(py::hasattr(scope, "__qualname__"))
		return kj::str(scope.attr("__qualname__").cast<kj::String>(), ".", name);
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
py::object interpretSchema(capnp::SchemaLoader& loader, uint64_t id, py::object scope);

py::object interpretStructSchema(capnp::SchemaLoader& loader, capnp::StructSchema schema, py::object scope) {	
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	auto structName = sanitizedStructName(schema.getUnqualifiedName());
	
	py::module_ collections = py::module_::import("collections");
	py::type mappingAbstractBaseClass = collections.attr("abc").attr("Mapping");
	
	py::dict attrs;
	attrs["__qualname__"] = qualName(scope, structName);
	attrs["__module__"] = moduleName;
	
	attrs["__init__"] = py::cpp_function([]() {
		KJ_UNIMPLEMENTED("Do not create instances of this class. Use StructType.newMessage() instead");
	});
	
	py::object output = (*baseMetaType)(structName, py::make_tuple(), attrs);
	
	// Create Builder, Reader, and Pipeline classes
	for(int i = 0; i < 3; ++i) {
		FSCPyClassType classType = (FSCPyClassType) i;
		
		py::dict attributes;
		for(StructSchema::Field field : schema.getFields()) {
			kj::StringPtr rawName = field.getProto().getName()
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
					attributes[str("init", nameUpper).cStr()] =  methodDescriptor(py::cpp_function(
						[field](DynamicStruct::Builder builder, size_t n) { return builder.init(field.getProto().getName(), n); }
					));
				}
			}
			
			attributes[name.cStr()] = field;
		}
		
		for(StructSchema::Field field : schema.getUnionFields()) {
			kj::StringPtr rawName = field.getProto().getName()
			kj::String name = memberName(rawName);
			
			using Field = capnp::schema::Field;
			
			auto type = field.getType();
			if(classType == FSCPyClassType::BUILDER) {
			
				kj::String nameUpper = kj::heapString(rawName);
				nameUpper[0] = toupper(name[0]);
				
				if(type.isStruct()) {
					attributes[str("init", nameUpper).cStr()] =  methodDescriptor(py::cpp_function(
						[field](DynamicStruct::Builder builder) { return DynamicValue::Builder(builder.init(field.getProto().getName())); }
					));
				}
			}
		}
		
		// Determine base class and class suffix
		py::type baseClass = py::type::of(py::none());
		kj::StringPtr suffix;
		
		switch(classType) {
			case FSCPyClassType::BUILDER: 
				baseClass = py::type::of<DynamicStruct::Builder>();
				suffix = "Builder";
				break;
				
			case FSCPyClassType::READER :
				baseClass = py::type::of<DynamicStruct::Reader>();
				suffix = "Reader";
				break;
				
			case FSCPyClassType::PIPELINE:
				baseClass = py::type::of<DynamicStructPipeline>();
				suffix = "Pipeline";
				break;  
		}
		
		attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
			[baseClass](py::object self, py::args args, py::kwargs kwargs) {
				baseClass.attr("__init__")(self, *args, **kwargs);
			}
		));
		
		// Determine metaclass and build a new type with the given suffix
		py::type metaClass = py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject*>(&PyType_Type));//py::type::of(baseClass);
		
		attributes["__qualname__"] = qualName(output, suffix);
		attributes["__module__"] = moduleName;
		
		// attributes["__slots__"] = py::make_tuple("_msg");
			
		py::object newCls = (*baseMetaType)(kj::str(structName, ".", suffix).cStr(), py::make_tuple(baseClass /*, mappingAbstractBaseClass*/), attributes);
				
		output.attr(suffix.cStr()) = newCls;
	}
	
	// Create Promise class
	{
		py::dict attributes;
		
		py::type promiseBase = py::type::of<PyPromise>();
		py::type pipelineBase = output.attr("Pipeline");
		
		attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
			[promiseBase, pipelineBase](py::object self, PyPromise& pyPromise, py::object pipeline) {
				promiseBase.attr("__init__")(self, pyPromise);
				pipelineBase.attr("__init__")(self, pipeline);
			}
		));
		
		kj::String suffix = kj::str("Promise");
		attributes["__qualname__"] = qualName(output, suffix);
		attributes["__module__"] = moduleName;
		
		//py::type metaClass = py::type::of(promiseBase);
		py::type metaClass = py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject*>(&PyType_Type));
		
		py::object newCls = (*baseMetaType)(kj::str(structName, ".", suffix).cStr(), py::make_tuple(promiseBase, pipelineBase), attributes);
		output.attr(suffix.cStr()) = newCls;
	}
		
	output.attr("newMessage") = py::cpp_function(
		[schema]() mutable {
			auto msg = new capnp::MallocMessageBuilder();
			
			// We use DynamicValue instead of DynamicStruct to engage our type-dependent dispatch
			capnp::DynamicValue::Builder builder = msg->initRoot<capnp::DynamicStruct>(schema);			
			py::object result = py::cast(builder);
			
			result.attr("_msg") = py::cast(msg, py::return_value_policy::take_ownership);
			
			return result;
		},
		py::name("newMessage"),
		py::scope(output)
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
	);
	
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr rawName = field.getProto().getName()
		kj::String name = memberName(rawName);
		
		using Field = capnp::schema::Field;
		
		switch(field.getProto().which()) {
			case Field::GROUP: {
				capnp::StructSchema subType = field.getType().asStruct();
				output.attr(name.cStr()) = interpretSchema(loader, subType.getProto().getId(), output);
				break;
			}
			
			case Field::SLOT: {} // Already processed above
		}
	}
	
	return output;
}

py::object interpretInterfaceSchema(capnp::SchemaLoader& loader, capnp::InterfaceSchema schema, py::object scope) {		
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	auto methods = schema.getMethods();
	
	auto collections = py::module_::import("collections");
	auto orderedDict = collections.attr("OrderedDict");
	
	// We need dynamic resolution to get our base capability client
	// Static resolution fails as we have overridden the type caster
	py::object baseObject = py::cast(DynamicCapability::Client());
	py::object baseClass = py::type::of(baseObject);
	
	py::type metaClass = py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject*>(&PyType_Type));
	//py::type metaClass = py::type::of(baseClass);
	
	py::dict outerAttrs;
	py::dict clientAttrs;
		
	py::object methodHolder = simpleObject();
	methodHolder.attr("__name__") = "methods";
	methodHolder.attr("__qualname__") = qualName(scope, str(schema.getUnqualifiedName(), ".methods"));
	methodHolder.attr("__module__") = moduleName;
	outerAttrs["methods"] = methodHolder;
	
	for(size_t i = 0; i < methods.size(); ++i) {
		// auto name = method.getProto().getName();
		auto method = methods[i];
		
		kj::StringPtr rawName = method.getProto().getName()
		kj::String name = memberName(rawName);
		
		auto paramType = method.getParamType();
		auto resultType = method.getResultType();
		
		auto doc = kj::strTree();
		
		kj::Vector<kj::StringTree> argumentDescs;
		kj::Vector<capnp::Type> types;
		kj::Vector<capnp::StructSchema::Field> argFields;
		
		size_t nArgs = 0;
				
		for(auto field : paramType.getNonUnionFields()) {
			// Only process slot fields
			if(field.getProto().which() != capnp::schema::Field::SLOT)
				continue;
			
			auto slot = field.getProto().getSlot();
			
			auto name = field.getProto().getName();
			auto type = field.getType();
			
			argumentDescs.add(strTree(typeName(type), " ", name));
			types.add(type);
			argFields.add(field);
		}
		
		auto isGeneratedStruct = [](capnp::Type type) {
			if(!type.isStruct())
				return false;
			
			auto asStruct = type.asStruct();
			auto displayName = asStruct.getProto().getDisplayName();
			
			KJ_IF_MAYBE(dontCare, displayName.findFirst('$')) {
				return true;
			} else {
				return false;
			}
		};
		
		auto paramName = isGeneratedStruct(paramType) ? strTree(name, ".", typeName(paramType)) : typeName(paramType);
		auto resultName = isGeneratedStruct(resultType) ? strTree(name, ".", typeName(resultType)) : typeName(resultType);
		
		auto functionDesc = kj::strTree(
			name, " : (", kj::StringTree(argumentDescs.releaseAsArray(), ", "), ") -> ", resultName.flatten(), ".Promise\n",
			"\n",
			"    or alternatively \n",
			"\n",
			name, " : (", paramName.flatten(), ") -> ", resultName.flatten(), ".Promise"
		);
		
		auto function = [paramType, resultType, method, types = mv(types), argFields = mv(argFields), nArgs](capnp::DynamicCapability::Client self, py::args pyArgs, py::kwargs pyKwargs) mutable {			
			// auto request = self.newRequest(method);
			capnp::Request<AnyPointer, AnyPointer> request = self.typelessRequest(
				method.getContainingInterface().getProto().getId(), method.getOrdinal(), nullptr
			);
			
			// Check whether we got the argument structure passed
			// In this case, copy the fields over from input struct
			
			bool requestBuilt = false;
			
			if(py::len(pyKwargs) == 0 && py::len(pyArgs) == 1) {
				// Check whether the first argument has the correct type
				auto argVal = pyArgs[0].cast<DynamicValue::Reader>();
				
				if(argVal.getType() == DynamicValue::STRUCT) {
					DynamicStruct::Reader asStruct = argVal.as<DynamicStruct>();
					
					if(asStruct.getSchema() == paramType) {
						request.setAs<DynamicStruct>(asStruct);
						
						requestBuilt = true;
					}
				}
			}

			if(!requestBuilt) {
				DynamicStruct::Builder structRequest = request.initAs<DynamicStruct>(paramType);
				// Parse positional arguments
				KJ_REQUIRE(pyArgs.size() <= argFields.size(), "Too many arguments specified");
				
				for(size_t i = 0; i < pyArgs.size(); ++i) {
					auto field = argFields[i];
					
					structRequest.set(field, pyArgs[i].cast<DynamicValue::Reader>());
				}
				
				// Parse keyword arguments
				auto processEntry = [&structRequest, paramType](kj::StringPtr name, DynamicValue::Reader value) mutable {
					KJ_IF_MAYBE(pField, paramType.findFieldByName(name)) {
						structRequest.set(*pField, value);
					} else {
						KJ_FAIL_REQUIRE("Unknown named parameter", name);
					}
				};
				auto pyProcessEntry = py::cpp_function(processEntry);
				
				auto nameList = py::list(pyKwargs);
				for(auto name : nameList) {
					pyProcessEntry(name, pyKwargs[name]);
				}
			}
			
			RemotePromise<AnyPointer> result = request.send();
			
			// Extract promise
			PyPromise resultPromise = result.then([resultType](capnp::Response<AnyPointer> response) {
				py::gil_scoped_acquire withGIL;
				
				DynamicValue::Reader structReader = response.getAs<DynamicStruct>(resultType);
				py::object pyReader = py::cast(structReader);
				
				py::object pyResponse = py::cast(mv(response));
				pyReader.attr("_response") = pyResponse;
				
				return kj::refcounted<PyObjectHolder>(mv(pyReader));
			});
			
			// Extract pipeline
			AnyPointer::Pipeline resultPipelineTypeless = mv(result);
			py::object pyPipeline = py::cast(DynamicStructPipeline(mv(resultPipelineTypeless), resultType));
			
			// Check for PromiseForResult class for type
			auto id = resultType.getProto().getId();
			
			// If not found, we can only return the promise for a generic result (perhaps once in the future
			// we can add the option for a full pass-through into RemotePromise<DynamicStruct>)
			py::object resultObject = py::cast(mv(resultPromise));
			
			if(globalClasses->contains(id)) {
				// Construct merged promise / pipeline object from PyPromise and pipeline
				py::type resultClass = (*globalClasses)[py::cast(id)].attr("Promise");
				
				resultObject = resultClass(resultObject, pyPipeline);
			} else {
				py::print("Could not find ID", id, "in global classes");
			}
			
			PromiseHandle handle;
			handle.pyPromise = resultObject;
			return handle;
		};
		
		auto pyFunction = py::cpp_function(
			kj::mv(function),
			py::return_value_policy::move,
			py::doc(functionDesc.flatten().cStr()),
			py::arg("self")
		);
				
		// TODO: Override the signature object
		
		py::object descriptor = methodDescriptor(pyFunction);
		descriptor.attr("__name__") = name.cStr();
		descriptor.attr("__module__") = moduleName;
		clientAttrs[name.cStr()] = descriptor;
		
		py::object holder = simpleObject();
		holder.attr("__qualname__") = qualName(scope, str(schema.getUnqualifiedName(), ".methods.", name));
		holder.attr("__name__") = name;
		holder.attr("__module__") = moduleName;
		holder.attr("desc") = str("Auxiliary classes (Params and/or Results) for method ", name);
		
		py::object pyParamType = interpretSchema(loader, method.getParamType().getProto().getId(), holder);
		py::object pyResultType = interpretSchema(loader, method.getResultType().getProto().getId(), holder);
		
		KJ_REQUIRE(!pyParamType.is_none());
		KJ_REQUIRE(!pyResultType.is_none());
		
		holder.attr(pyParamType.attr("__name__")) = pyParamType;
		holder.attr(pyResultType.attr("__name__")) = pyResultType;
		
		methodHolder.attr(name.cStr()) = holder;
	}
	
	outerAttrs["newDeferred"] = py::cpp_function(
		[schema](PyPromise promise) {
			auto untypedPromise = promise.as<capnp::DynamicCapability::Client>()
			.then([](capnp::DynamicCapability::Client typed) mutable -> capnp::Capability::Client {
				return mv(typed);
			});
			
			capnp::Capability::Client untyped = mv(untypedPromise);
			
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
	
	/*py::print("Extracting surrounding module");
	py::module_ module_ = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope;*/
	
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
	
	
	return outerCls;
}

py::object interpretSchema(capnp::SchemaLoader& loader, uint64_t id, py::object scope) {	
	if(globalClasses->contains(id))
		return (*globalClasses)[py::cast(id)];
	
	KJ_IF_MAYBE(dontCare, loader.tryGet(id)) {
	} else {
		return py::none();
	}
	
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	
	capnp::Schema schema = loader.get(id);
	
	py::object output = py::none();
	
	switch(schema.getProto().which()) {
		case capnp::schema::Node::STRUCT:
			output = interpretStructSchema(loader, schema.asStruct(), scope);
			break;
		case capnp::schema::Node::INTERFACE:
			output = interpretInterfaceSchema(loader, schema.asInterface(), scope);
			break;
		
		default:
			py::dict attrs;
			attrs["__qualname__"] = qualName(scope, schema.getUnqualifiedName()).asPtr();
			attrs["__module__"] = moduleName;
			
			output = (*baseMetaType)(schema.getUnqualifiedName().cStr(), py::make_tuple(), attrs);
			break;
	}
	
	if(output.is_none())
		return output;
	
	// Interpret child objects
	for(auto nestedNode : schema.getProto().getNestedNodes()) {
		py::object subObject = interpretSchema(loader, nestedNode.getId(), output);
		
		if(subObject.is_none())
			continue;
			
		output.attr(nestedNode.getName().cStr()) = subObject;
	}
	
	(*globalClasses)[py::cast(id)] = output;
	
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

// ================== Brand conversion helpers ============================

void fscpy::extractType(capnp::Type in, capnp::schema::Type::Builder out) {	
	switch(in.which()) {
		#define HANDLE_VALUE(enumVal, name) \
			case capnp::schema::Type::enumVal: {\
				auto outTyped = out.init ## name(); \
				auto inTyped  = in.as ## name(); \
				outTyped.setTypeId(inTyped.getProto().getId()); \
				extractBrand(inTyped, outTyped.initBrand()); \
				break; \
			}
			
		HANDLE_VALUE(ENUM, Enum);
		HANDLE_VALUE(INTERFACE, Interface);
		HANDLE_VALUE(STRUCT, Struct);
		
		#undef HANDLE_VALUE
		
		case capnp::schema::Type::LIST: {
			auto outAsList = out.initList();
			extractType(in.asList().getElementType(), outAsList.getElementType());
			break;
		}
		case capnp::schema::Type::ANY_POINTER: {
			auto outAsAny = out.initAnyPointer();
			
			KJ_IF_MAYBE(pBrandParameter, in.getBrandParameter()) {
				auto outAsParam = outAsAny.initParameter();
				outAsParam.setScopeId(pBrandParameter->scopeId);
				outAsParam.setParameterIndex(pBrandParameter->index);
				break;
			}
			
			KJ_IF_MAYBE(pImplicitParameter, in.getImplicitParameter()) {
				auto outAsParam = outAsAny.initImplicitMethodParameter();
				outAsParam.setParameterIndex(pImplicitParameter->index);
				break;
			}
			
			auto outAsUnconstrained = outAsAny.initUnconstrained();
			switch(in.whichAnyPointerKind()) {
				#define HANDLE_VALUE(enumVal, name) \
					case capnp::schema::Type::AnyPointer::Unconstrained::enumVal: { \
						outAsUnconstrained.set ## name(); \
						break; \
					}
				
				HANDLE_VALUE(ANY_KIND, AnyKind);
				HANDLE_VALUE(STRUCT, Struct);
				HANDLE_VALUE(LIST, List);
				HANDLE_VALUE(CAPABILITY, Capability);
				
				#undef HANDLE_VALUE
			}
			
			break;
		}
		
		#define HANDLE_VALUE(enumVal, name) \
		case capnp::schema::Type::enumVal: \
			out.set ## name(); \
			break;
			
		HANDLE_VALUE(VOID, Void);
		HANDLE_VALUE(BOOL, Bool);
		
		HANDLE_VALUE(INT8, Int8);
		HANDLE_VALUE(INT16, Int16);
		HANDLE_VALUE(INT32, Int32);
		HANDLE_VALUE(INT64, Int64);
		
		HANDLE_VALUE(UINT8, Uint8);
		HANDLE_VALUE(UINT16, Uint16);
		HANDLE_VALUE(UINT32, Uint32);
		HANDLE_VALUE(UINT64, Uint64);
		
		HANDLE_VALUE(FLOAT32, Float32);
		HANDLE_VALUE(FLOAT64, Float64);
		
		HANDLE_VALUE(TEXT, Text);
		HANDLE_VALUE(DATA, Data);
		
		#undef HANDLE_VALUE
	}
}

void fscpy::extractBrand(capnp::Schema in, capnp::schema::Brand::Builder out) {
	if(!in.getProto().getIsGeneric()) {
		out.initScopes(0);
		return;
	}
	
	auto scopeIds = in.getGenericScopeIds();
	
	auto outScopes = out.initScopes(scopeIds.size());
	for(auto iScope : kj::indices(scopeIds)) {
		auto outScope = outScopes[iScope];
		outScope.setScopeId(scopeIds[iScope]);
		
		auto inBindings  = in.getBrandArgumentsAtScope(scopeIds[iScope]);
		auto outBindings = outScope.initBind(inBindings.size());
		for(auto iBinding : kj::indices(outBindings)) {
			capnp::Type inType = inBindings[iBinding];
			auto outBinding = outBindings[iBinding];
			
			extractType(inType, outBinding.initType());
		}
	}
}

// ================== Implementation of fscpy::Loader =====================

bool fscpy::Loader::importNode(uint64_t nodeID, py::module scope) {
	auto obj = interpretSchema(capnpLoader, nodeID, scope);
	auto schema = capnpLoader.get(nodeID);
	
	if(!obj.is_none()) {
		if(py::hasattr(scope, schema.getUnqualifiedName().cStr()))
			return false;
		
		scope.add_object(schema.getUnqualifiedName().cStr(), obj);
		return true;
	}
	
	return false;
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

void fscpy::initLoader(py::module_& m) {}