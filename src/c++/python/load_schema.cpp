#include "fscpy.h"

#include <capnp/dynamic.h>
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

namespace {

enum class FSCPyClassType {
	BUILDER, READER, PIPELINE
};

py::object interpretStructSchema(capnp::SchemaLoader& loader, capnp::StructSchema schema) {
	KJ_LOG(WARNING, "Interpreting struct schema");
	py::object output = (*baseMetaType)(schema.getUnqualifiedName().cStr(), py::make_tuple(), py::dict());
	
	for(int i = 0; i < 3; ++i) {
		FSCPyClassType classType = (FSCPyClassType) i;
		KJ_LOG(WARNING, "Building subclass", i);
		
		py::dict attributes;
		for(StructSchema::Field field : schema.getFields()) {
			kj::StringPtr name = field.getProto().getName();
			KJ_LOG(WARNING, "Processing field", name);
			
			using Field = capnp::schema::Field;
			
			switch(field.getProto().which()) {
				case Field::SLOT: {
					KJ_LOG(WARNING, "Is slot");
					
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
		
		py::type metaClass = py::type::of(baseClass);
		
		kj::String suffix;
		
		switch(classType) {
			case FSCPyClassType::BUILDER:  suffix = kj::str("Builder");  break;
			case FSCPyClassType::READER :  suffix = kj::str("Reader");   break;
			case FSCPyClassType::PIPELINE: suffix = kj::str("Pipeline"); break; 
		}
			
		py::object newCls = metaClass(kj::str(schema.getUnqualifiedName(), ".", suffix).cStr(), py::make_tuple(baseClass), attributes);
		output.attr(suffix.cStr()) = newCls;
	}
	
	KJ_LOG(WARNING, "Building groups");
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr name = field.getProto().getName();
		
		using Field = capnp::schema::Field;
		
		switch(field.getProto().which()) {
			case Field::GROUP: {
				KJ_LOG(WARNING, "Building group", name);
				capnp::StructSchema subType = field.getType().asStruct();
				output.attr(name.cStr()) = interpretStructSchema(loader, subType);
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
			return strTree(asStruct.getUnqualifiedName());
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

py::object interpretInterfaceSchema(capnp::SchemaLoader& loader, capnp::InterfaceSchema schema) {	
	auto methods = schema.getMethods();
	
	auto collections = py::module_::import("Collections");
	auto orderedDict = collections.attr("OrderedDict");
	
	py::dict attributes;
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
			name, " : ", kj::StringTree(argumentDescs.releaseAsArray(), ", "), " -> ", typeName(resultType), "\n",
			" or alternatively \n",
			name, " : ", typeName(paramType), " -> ", typeName(resultType)
		);
		
		auto function = [paramType, resultType, types = mv(types), argFields = mv(argFields), nArgs](capnp::DynamicCapability::Client self, py::args pyArgs, py::kwargs pyKwargs) mutable {			
			// Untyped function body
			auto body = [paramType, resultType](DynamicStruct::Reader arg) {
				KJ_REQUIRE(arg.getSchema() == paramType, "Invalid type, expected", paramType, "but got", arg.getSchema());
				
				KJ_UNIMPLEMENTED();
				return DynamicStruct::Reader();
			};
			
			// Check whether we got the argument structure passed
			// In this case, just run the function body directly
			
			if(!pyKwargs && py::len(pyArgs) == 1) {
				// Check whether the first argument has the correct type
				auto argVal = pyArgs[0].cast<DynamicValue::Reader>();
				
				if(argVal.getType() == DynamicValue::STRUCT) {
					DynamicStruct::Reader asStruct = argVal.as<DynamicStruct>();
					
					if(asStruct.getSchema() == paramType)
						return body(asStruct);
				}
			}

			// Parse positional arguments
			KJ_REQUIRE(pyArgs.size() <= argFields.size(), "Too many arguments specified");
			
			fsc::Temporary<DynamicStruct> newArg(paramType);
			for(size_t i = 0; i < pyArgs.size(); ++i) {
				auto field = argFields[i];
				
				newArg.set(field, pyArgs[i].cast<DynamicValue::Reader>());
			}
			
			// Parse keyword arguments
			auto processEntry = [&newArg, paramType](kj::StringPtr name, DynamicValue::Reader value) mutable {
				KJ_IF_MAYBE(pField, paramType.findFieldByName(name)) {
					newArg.set(*pField, value);
				} else {
					KJ_FAIL_REQUIRE("Unknown named parameter", name);
				}
			};
			auto pyProcessEntry = py::cpp_function(processEntry);
			
			auto nameList = py::list(pyKwargs);
			for(auto name : nameList) {
				pyProcessEntry(name, pyKwargs[name]);
			}
			
			return body(newArg.asReader());
		};
		
		auto pyFunction = py::cpp_function(
			kj::mv(function),
			py::doc(functionDesc.flatten().cStr())
		);
				
		// TODO: Override the signature object
		
		attributes[name.cStr()] = pyFunction;
	}
	
	// We need dynamic resolution to get our base capability client
	// Static resolution fails as we have overridden the type caster
	py::object baseObject = py::cast(DynamicCapability::Client());
	py::object baseClass = py::type::of(baseObject);
	
	py::type metaClass = py::type::of(baseClass);	
	py::object newCls = metaClass(schema.getUnqualifiedName(), py::tuple(baseClass), attributes);
	
	py::object output = (*baseMetaType)(schema.getUnqualifiedName().cStr(), py::make_tuple(), py::dict());
	output.attr("Client") = newCls;
	
	return output;
}

py::object interpretSchema(capnp::SchemaLoader& loader, uint64_t id) {	
	if(globalClasses->contains(id))
		return (*globalClasses)[py::cast(id)];
	
	KJ_IF_MAYBE(dontCare, loader.tryGet(id)) {
	} else {
		return py::none();
	}
	
	Schema schema = loader.get(id);
	
	py::object output = py::none();
	
	switch(schema.getProto().which()) {
		case capnp::schema::Node::STRUCT:
			output = interpretStructSchema(loader, schema.asStruct());
			break;
		case capnp::schema::Node::INTERFACE:
			output = interpretInterfaceSchema(loader, schema.asInterface());
			break;
		
		default:
			output = (*baseMetaType)(schema.getUnqualifiedName().cStr(), py::make_tuple(), py::dict());
			break;
	}	
	
	// Interpret child objects
	for(auto nestedNode : schema.getProto().getNestedNodes()) {
		py::object subObject = interpretSchema(loader, nestedNode.getId());
		
		if(subObject.is_none())
			continue;
		
		output.attr(nestedNode.getName().cStr()) = subObject;
	}
	
	(*globalClasses)[py::cast(id)] = output;
	
	return output;
}

capnp::SchemaLoader defaultLoader;

}

namespace fscpy {

void loadDefaultSchema(py::module_& m) {
	using capnp::SchemaLoader;
	
	defaultLoader.loadCompiledTypeAndDependencies<capnp::schema::Node>();
	
	auto m2 = m.def_submodule("gen");
	m2.attr("__all__") = py::list();
	
	py::list m2all = m2.attr("__all__");
	
	for(auto schema : defaultLoader.getAllLoaded()) {
		auto parentId = schema.getProto().getScopeId();
		
		KJ_IF_MAYBE(dontCare, defaultLoader.tryGet(parentId)) {
		} else {
			auto name = schema.getUnqualifiedName();
			KJ_LOG(WARNING, "Interpreting root node", name);
			
			auto obj = interpretSchema(defaultLoader, schema.getProto().getId());
			
			if(obj.is_none())
				continue;
			
			m2.add_object(name.cStr(), obj);
			m2all.append(name.cStr());
		}
	}
}

}