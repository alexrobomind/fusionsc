#pragma once

#include "fscpy.h"

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/schema.h>

namespace fscpy {

//! Conversion helper to obtain DynamicValue from Python and NumPy scalar types
Maybe<capnp::DynamicValue::Reader> dynamicValueFromScalar(py::handle handle);

struct DynamicStructPipeline;

struct DynamicValuePipeline {
	capnp::AnyPointer::Pipeline typeless;
	capnp::Schema schema;
	
	inline DynamicValuePipeline(capnp::AnyPointer::Pipeline typeless, capnp::Schema schema) :
		typeless(mv(typeless)),
		schema(mv(schema))
	{}
	
	inline DynamicValuePipeline() : typeless(nullptr), schema() {}
	
	inline DynamicValuePipeline(DynamicValuePipeline& other) :
		typeless(other.typeless.noop()),
		schema(other.schema)
	{}
	
	inline DynamicValuePipeline(DynamicValuePipeline&& other) = default;
	
	DynamicStructPipeline asStruct();
	capnp::DynamicCapability::Client asCapability();
};

struct DynamicStructPipeline {
	capnp::AnyPointer::Pipeline typeless;
	capnp::StructSchema schema;
	
	inline DynamicStructPipeline(capnp::AnyPointer::Pipeline typeless, capnp::StructSchema schema) :
		typeless(mv(typeless)),
		schema(mv(schema))
	{}
	
	inline DynamicStructPipeline() : typeless(nullptr), schema() {}
	
	inline DynamicStructPipeline(DynamicStructPipeline& other) :
		typeless(other.typeless.noop()),
		schema(other.schema)
	{}
	
	inline DynamicStructPipeline(DynamicStructPipeline&& other) = default;
	
	inline capnp::StructSchema getSchema() { return schema; }
	
	DynamicValuePipeline get(capnp::StructSchema::Field field);
	DynamicValuePipeline get(kj::StringPtr fieldName);
};

}