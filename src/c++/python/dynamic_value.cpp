#include "dynamic_value.h"

#include <capnp/dynamic.h>
#include <capnp/schema.h>
#include <capnp/schema-loader.h>

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
}

py::object interpretSchema(capnp::SchemaLoader& loader, uint64_t id);

py::object interpretStructSchema(capnp::SchemaLoader& loader, capnp::StructSchema schema) {
	py::object output;
	
	for(int i = 0; i < 3; ++i) {
		FSCPyClassType classType = (FSCPyClassType) i;
		
		// Determine base class and metaclass
		py::type baseClass = py::type::of(py::none());
		
		switch(classType) {
			case FSCPyClassType::BUILDER:  baseClass = py::type::of<DynamicStruct::Builder>();  break;
			case FSCPyClassType::READER :  baseClass = py::type::of<DynamicStruct::Reader>();   break;
			case FSCPyClassType::PIPELINE: baseClass = py::type::of<DynamicStruct::Pipeline>(); break; 
		}
		
		py::type metaClass = py::type::of(baseClass);
		
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

py::object interpretSchema(capnp::SchemaLoader& loader, uint64_t id) {
	if(globalClasses->contains(id))
		return (*globalClasses)[py::cast(id)];
	
	Schema schema = loader.get(id);
	
	py::object output = py::none();
	
	switch(schema.getProto().which()) {
		case capnp::schema::Node::STRUCT:
			output = interpretStructSchema(loader, schema.asStruct());
	}
	
	(*globalClasses)[py::cast(id)] = output;
	return output;
}

}