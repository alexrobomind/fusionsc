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
			return defaultLoader.capnpLoader.get(id);
		}
	);
		
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
