#include "fscpy.h"

#include <capnp/dynamic.h>
#include <capnp/schema.h>
#include <capnp/schema-loader.h>

#include <kj/string-tree.h>

kj::Own<py::dict> globalClasses;

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

struct ConstFieldDescriptor {
	StructSchema::Field field;
	
	ConstFieldDescriptor(StructSchema::Field field) : field(field) {}
	
	DynamicValue::Reader get(DynamicStruct::Reader& self, py::object type) { return self.get(field); }
};

struct FieldDescriptor {
	StructSchema::Field field;
	
	FieldDescriptor(StructSchema::Field field) : field(field) {}
	
	DynamicValue::Builder get(DynamicStruct::Builder& self, py::object type) { return self.get(field); }
	void set(DynamicStruct::Builder& self, DynamicValue::Reader value) { self.set(field, value);	}
	void del(DynamicStruct::Builder& self) { self.clear(field); }
};

py::type baseType;

}

namespace fscpy {

void dynamicValueBindings(py::module_& m) {
	py::class_<capnp::StructSchema::Field>(m, "Field")
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Pipeline& self, py::object type) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Reader& self, py::object type) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__get__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, py::object type) { return self.get(field); },
			py::arg("obj"), py::arg("type") = py::none()
		)
		.def("__set__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self, DynamicValue::Reader value) { self.set(field, value); })
		.def("__delete__", [](capnp::StructSchema::Field& field, DynamicStruct::Builder& self) { self.clear(field); })
	;
	
	baseType = py::eval("type('CapnpStructureNode', (object,), {})");
}

void loadDefaultSchemas(py::module_& m) {
}

py::object interpretStructSchema(capnp::SchemaLoader& loader, capnp::StructSchema schema) {
	py::object output = baseType();
	
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
					
					attributes[py::cast(name)] = field;
					break;
				}
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
			
		py::object newCls = metaClass(schema.getUnqualifiedName(), py::tuple(baseClass), attributes);
		
		switch(classType) {
			case FSCPyClassType::BUILDER:  output.attr("Builder")  = newCls;  break;
			case FSCPyClassType::READER :  output.attr("Reader")   = newCls;  break;
			case FSCPyClassType::PIPELINE: output.attr("Pipeline") = newCls; break; 
		}
	}
	
	for(StructSchema::Field field : schema.getFields()) {
		kj::StringPtr name = field.getProto().getName();
		
		using Field = capnp::schema::Field;
		
		switch(field.getProto().which()) {
			case Field::GROUP: {
				capnp::StructSchema subType = field.getType().asStruct();
				output.attr(py::cast(name)) = interpretStructSchema(loader, subType);
				break;
			}
		}
	}
	
	return output;
}

py::object interpretInterfaceSchema(capnp::SchemaLoader& loader, capnp::InterfaceSchema schema) {	
	auto methods = schema.getMethods();
	
	auto collections = py::module_::import("Collections");
	auto orderedDict = collections.attr("OrderedDict");
	
	py::dict attributes;
	for(size_t i = 0; i < methods.size(); ++i) {
		auto method = methods[i];
		auto name = method.getName();
		
		auto paramType = method.getParamType();
		auto resultType = method.getResultType();
		
		auto doc = strTree();
		
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
			
			argumentDescs.add(strTree(type, " ", name));
			types.add(type);
			argNames.add(name);
			
			nArgs++;
		}
		
		auto functionDesc = strTree(
			name, " : " kj::StringTree(argumentDescs.releaseAsArray(), ", "), " -> ", resultType, "\n",
			" or alternatively \n",
			name, " : ", paramType, " -> ", resultType
		);
		
		auto function = [paramType, resultType, types = mv(types), argFields = mv(argFields), nArgs](capnp::DynamicCapability::Client self, py::args pyArgs, py::kwargs pyKwargs) mutable {			
			// Untyped function body
			auto body = [paramType, resultType](DynamicValue::Reader arg) {
				KJ_REQUIRE(arg.getType() == paramType, "Invalid type, expected", paramType, "but got", arg.getType());
			}
			
			// Check whether we got the argument structure passed
			// In this case, just run the function body directly
			
			if(!pyKwargs && py::len(pyArgs) == 1) {
				// Check whether the first argument has the correct type
				auto argVal = pyArgs[0].cast<DynamicValue::Reader>();
				
				if(argVal.getType() == paramType)
					return body(argVal);				
			}

			// Parse positional arguments
			KJ_REQUIRE(pyArgs.size() <= argFields.size(), "Too many arguments specified");
			
			Temporary<DynamicValue> newArg(paramType);
			for(size_t i = 0; i < pyArgs.size(); ++i) {
				auto field = argFields[i];
				
				newArg.set(field, pyArgs[i].cast<DynamicValue::Reader>());
			}
			
			// Parse keyword arguments
			auto processEntry = [&newArg, paramType](const kj::StringPtr name, DynamicValue::Reader value) {
				KJ_IF_MAYBE(pField, paramType.findFieldByName(name)) {
					newArg.set(*pField, value);
				} else {
					KJ_REQUIRE_FAIL("Unknown named parameter", name);
				}
			};
			auto pyProcessEntry = py::cpp_function(processEntry);
			
			auto nameList = py::list(kwArgs);
			for(auto name : nameList) {
				pyProcessEntry(name, pyKwargs[name]);
			}
			
			return body(newArg);
		};
		
		auto pyFunction = py::cpp_function(
			function,
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
	
	py::object output = baseType();
	output.attr("Client") = newCls;
}

py::object interpretSchema(capnp::SchemaLoader& loader, uint64_t id) {
	if(globalClasses->contains(id))
		return (*globalClasses)[py::cast(id)];
	
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
			return py::none();
	}
	
	// Interpret child objects
	for(auto nestedNode : schema.getProto().getNestedNodes()) {
		py::object subObject = interpretSchema(loader, nestedNode.getId());
		
		if(subObject == py::none())
			continue;
		
		output.attr(nestedNode.getName().cStr()) = subObject;
	}
	
	(*globalClasses)[py::cast(id)] = output;
	return output;
}

}