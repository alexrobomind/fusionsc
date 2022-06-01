#include "fscpy.h"
#include "async.h"

#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/schema-loader.h>

#include <kj/string-tree.h>

#include <fsc/data.h>

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

using capnp::Schema;
using capnp::StructSchema;

namespace fscpy { namespace {

struct PromiseHandle {
	py::object pyPromise;
};

}}

namespace pybind11 { namespace detail {
	template<>
	struct type_caster<fscpy::PromiseHandle> {
		using PromiseHandle = fscpy::PromiseHandle;
		
		PYBIND11_TYPE_CASTER(fscpy::PromiseHandle, const_name("PromiseForResult"));
		
		bool load(handle src, bool convert) {
			return false;		
		}
		
		static handle cast(PromiseHandle src, return_value_policy policy, handle parent) {
			return src.pyPromise.inc_ref();
		}
	};
}}

namespace fscpy {

namespace {

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

py::object interpretStructSchema(capnp::SchemaLoader& loader, capnp::StructSchema schema, py::object scope) {	
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	auto structName = sanitizedStructName(schema.getUnqualifiedName());
	
	py::dict attrs;
	attrs["__qualname__"] = qualName(scope, structName);
	attrs["__module__"] = moduleName;
	
	py::object output = (*baseMetaType)(structName, py::make_tuple(), attrs);
	
	// Create Builder, Reader, and Pipeline classes
	for(int i = 0; i < 3; ++i) {
		FSCPyClassType classType = (FSCPyClassType) i;
		
		py::dict attributes;
		for(StructSchema::Field field : schema.getFields()) {
			kj::StringPtr name = field.getProto().getName();
			
			using Field = capnp::schema::Field;
			
			switch(field.getProto().which()) {
				case Field::SLOT: {					
					auto slot = field.getProto().getSlot();
					auto type = field.getType();
					
					// Only emit pipeline fields for struct and interface fields
					if(classType == FSCPyClassType::PIPELINE && !type.isStruct() && !type.isInterface())
						break;
					
					attributes[name.cStr()] = field;
					break;
				}
				
				case Field::GROUP: {} // We process these later
			}
		}
		
		// Determine base class and metaclass
		py::type baseClass = py::type::of(py::none());
		
		switch(classType) {
			case FSCPyClassType::BUILDER:  baseClass = py::type::of<DynamicStruct::Builder>();  break;
			case FSCPyClassType::READER :  baseClass = py::type::of<DynamicStruct::Reader>();   break;
			case FSCPyClassType::PIPELINE: baseClass = py::type::of<DynamicStruct::Pipeline>(); break; 
		}
		
		attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
			[baseClass](py::object self, py::args args, py::kwargs kwargs) {
				baseClass.attr("__init__")(self, *args, **kwargs);
			}
		));
		
		py::type metaClass = py::type::of(baseClass);
		
		kj::String suffix;
		switch(classType) {
			case FSCPyClassType::BUILDER:  suffix = kj::str("Builder");  break;
			case FSCPyClassType::READER :  suffix = kj::str("Reader");   break;
			case FSCPyClassType::PIPELINE: suffix = kj::str("Pipeline"); break; 
		}
		
		attributes["__qualname__"] = qualName(output, suffix);
		attributes["__module__"] = moduleName;
			
		py::object newCls = metaClass(kj::str(structName, ".", suffix).cStr(), py::make_tuple(baseClass), attributes);
		output.attr(suffix.cStr()) = newCls;
	}
	
	// Create Promise class
	{
		py::dict attributes;
		
		py::type promiseBase = py::type::of<PyPromise>();
		py::type pipelineBase = output.attr("Pipeline");
		
		attributes["__init__"] = fscpy::methodDescriptor(py::cpp_function(
			[promiseBase, pipelineBase](py::object self, PyPromise& pyPromise, py::object pipeline, py::object key) {
				promiseBase.attr("__init__")(self, pyPromise);
				pipelineBase.attr("__init__")(self, pipeline, key);
			}
		));
		
		kj::String suffix = kj::str("Promise");
		attributes["__qualname__"] = qualName(output, suffix);
		attributes["__module__"] = moduleName;
		
		py::type metaClass = py::type::of(promiseBase);
		
		py::object newCls = metaClass(kj::str(structName, ".", suffix).cStr(), py::make_tuple(promiseBase, pipelineBase), attributes);
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
	
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr name = field.getProto().getName();
		
		using Field = capnp::schema::Field;
		
		switch(field.getProto().which()) {
			case Field::GROUP: {
				capnp::StructSchema subType = field.getType().asStruct();
				output.attr(name.cStr()) = interpretStructSchema(loader, subType, output);
				break;
			}
			
			case Field::SLOT: {} // Already processed above
		}
	}
	
	return output;
}

kj::StringTree typeName(capnp::Type type) {
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

py::object interpretInterfaceSchema(capnp::SchemaLoader& loader, capnp::InterfaceSchema schema, py::object scope) {	
	py::print("Building class ", schema.getUnqualifiedName(), " in ", scope);
	
	py::str moduleName = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope.attr("__name__");
	auto methods = schema.getMethods();
	
	auto collections = py::module_::import("collections");
	auto orderedDict = collections.attr("OrderedDict");
	
	// We need dynamic resolution to get our base capability client
	// Static resolution fails as we have overridden the type caster
	py::object baseObject = py::cast(DynamicCapability::Client());
	py::object baseClass = py::type::of(baseObject);
	py::type metaClass = py::type::of(baseClass);
	
	py::dict outerAttrs;
	py::dict clientAttrs;
	
	for(size_t i = 0; i < methods.size(); ++i) {
		auto method = methods[i];
		auto name = method.getProto().getName();
		
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
			auto type = field.getType();;
			
			argumentDescs.add(strTree(typeName(type), " ", name));
			types.add(type);
			argFields.add(field);
			
			nArgs++;
		}
		
		auto functionDesc = kj::strTree(
			name, " : (", kj::StringTree(argumentDescs.releaseAsArray(), ", "), ") -> ", typeName(resultType), "\n",
			"\n",
			"    or alternatively \n",
			"\n",
			name, " : (", typeName(paramType), ") -> ", typeName(resultType)
		);
		
		auto function = [paramType, resultType, method, types = mv(types), argFields = mv(argFields), nArgs](capnp::DynamicCapability::Client self, py::args pyArgs, py::kwargs pyKwargs) mutable {
			auto request = self.newRequest(method);
			
			// Check whether we got the argument structure passed
			// In this case, copy the fields over from input struct
			
			bool requestBuilt = false;
			
			if(!pyKwargs && py::len(pyArgs) == 1) {
				// Check whether the first argument has the correct type
				auto argVal = pyArgs[0].cast<DynamicValue::Reader>();
				
				if(argVal.getType() == DynamicValue::STRUCT) {
					DynamicStruct::Reader asStruct = argVal.as<DynamicStruct>();
					
					if(asStruct.getSchema() == paramType) {
						for(auto field : paramType.getNonUnionFields()) {
							request.set(field, asStruct.get(field));
						}
						
						KJ_IF_MAYBE(pField, asStruct.which()) {
							request.set(*pField, asStruct.get(*pField));
						}
						
						requestBuilt = true;
					}
				}
			}

			if(!requestBuilt) {
				// Parse positional arguments
				KJ_REQUIRE(pyArgs.size() <= argFields.size(), "Too many arguments specified");
				
				for(size_t i = 0; i < pyArgs.size(); ++i) {
					auto field = argFields[i];
					
					request.set(field, pyArgs[i].cast<DynamicValue::Reader>());
				}
				
				// Parse keyword arguments
				auto processEntry = [&request, paramType](kj::StringPtr name, DynamicValue::Reader value) mutable {
					KJ_IF_MAYBE(pField, paramType.findFieldByName(name)) {
						request.set(*pField, value);
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
			
			
			using capnp::RemotePromise;
			using capnp::Response;
			
			RemotePromise<DynamicStruct> result = request.send();
			
			// Extract promise
			PyPromise resultPromise = result.then([](capnp::Response<capnp::DynamicStruct> response) {
				return py::cast(mv(response));
			});
			
			// Extract pipeline
			DynamicStruct::Pipeline resultPipeline = mv(result);
			py::object pyPipeline = py::cast(mv(resultPipeline));
			
			// Check for PromiseForResult class for type
			auto id = resultType.getProto().getId();
			
			// If not found, we can only return the promise for a generic result (perhaps once in the future
			// we can add the option for a full pass-through into RemotePromise<DynamicStruct>)
			py::object resultObject = py::cast(resultPromise);
			if(globalClasses->contains(id)) {
				// Construct merged promise / pipeline object from PyPromise and pipeline
				py::type resultClass = (*globalClasses[id]).attr("Promise");
				
				resultObject = resultClass(resultPromise, pyPipeline, INTERNAL_ACCESS_KEY);
			}
			
			PromiseHandle handle;
			handle.pyPromise = resultObject;
			return handle;
		};
		
		auto pyFunction = py::cpp_function(
			kj::mv(function),
			py::doc(functionDesc.flatten().cStr()),
			py::arg("self")
		);
		
		py::print("Assigning method", name.cStr());
				
		// TODO: Override the signature object
		
		py::object descriptor = methodDescriptor(pyFunction);
		descriptor.attr("__name__") = name.cStr();
		descriptor.attr("__module__") = moduleName;
		
		clientAttrs[name.cStr()] = descriptor;
		
		py::object holder = simpleObject();
		holder.attr("__qualname__") = qualName(scope, str(schema.getUnqualifiedName(), ".", name));
		holder.attr("__name__") = name;
		holder.attr("__module__") = moduleName;
		holder.attr("desc") = str("Auxiliary classes for method ", name);
		
		py::object pyParamType = interpretStructSchema(loader, method.getParamType(), holder);
		py::object pyResultType = interpretStructSchema(loader, method.getResultType(), holder);
		
		KJ_REQUIRE(!pyParamType.is_none());
		KJ_REQUIRE(!pyResultType.is_none());
		
		holder.attr(pyParamType.attr("__name__")) = pyParamType;
		holder.attr(pyResultType.attr("__name__")) = pyResultType;
		
		outerAttrs[name.cStr()] = holder;
	}
	
	/*py::print("Extracting surrounding module");
	py::module_ module_ = py::hasattr(scope, "__module__") ? scope.attr("__module__") : scope;*/
	
	py::print("Creating outer class");
	
	auto outerName = qualName(scope, schema.getUnqualifiedName());
	outerAttrs["__qualname__"] = outerName;
	outerAttrs["__module__"] = moduleName;
	
	py::object outerCls = metaClass(schema.getUnqualifiedName(), py::make_tuple(baseClass), outerAttrs);
	
	py::print("Creating client class");
	auto innerName = qualName(outerCls, "Client");
	
	clientAttrs["__qualname__"] = innerName;
	clientAttrs["__module__"] = moduleName;
	
	py::object clientCls = (*baseMetaType)(schema.getUnqualifiedName().cStr(), py::make_tuple(), clientAttrs);
	outerCls.attr("Client") = clientCls;
	
	py::print("Done");
	
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
	
	Schema schema = loader.get(id);
	
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

capnp::SchemaLoader defaultLoader;

}

void loadDefaultSchema(py::module_& m) {
	using capnp::SchemaLoader;
	
	defaultLoader.loadCompiledTypeAndDependencies<DataService>();
	
	//auto m2 = m.def_submodule("api");
	//m2.attr("__all__") = py::list();
	
	//py::list m2all = m2.attr("__all__");
	
	for(auto schema : defaultLoader.getAllLoaded()) {
		auto parentId = schema.getProto().getScopeId();
		
		KJ_IF_MAYBE(dontCare, schema.getUnqualifiedName().findFirst('$')) {
			continue;
		}
		
		py::print("Root object ", schema.getUnqualifiedName(), " with parentID ", parentId);
		
		KJ_IF_MAYBE(dontCare, defaultLoader.tryGet(parentId)) {
		} else {
			auto name = schema.getUnqualifiedName();
			
			auto obj = interpretSchema(defaultLoader, schema.getProto().getId(), m);
			
			if(obj.is_none())
				continue;
			
			m.add_object(name.cStr(), obj);
			//m2all.append(name.cStr());
		}
	}
}

}