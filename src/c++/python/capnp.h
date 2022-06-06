#include "fscpy.h"

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/schema.h>

namespace fscpy {

struct DynamicStructPipeline;

struct DynamicValuePipeline {
	capnp::AnyPointer::Pipeline typeless;
	capnp::Schema schema;
	
	inline DynamicValuePipeline(capnp::AnyPointer::Pipeline typeless, capnp::Schema schema) :
		typeless(mv(typeless)),
		schema(mv(schema))
	{}
	
	inline DynamicValuePipeline(DynamicValuePipeline& other) :
		typeless(other.typeless.noop()),
		
	
	DynamicStructPipeline asStruct();
	capnp::DynamicCapability::Client asCapability();
};

struct DynamicStructPipeline {
	capnp::AnyPointer::Pipeline typeless;
	capnp::StructSchema schema;
	
	inline DynamicValuePipeline(capnp::AnyPointer::Pipeline typeless, capnp::StructSchema schema) :
		typeless(mv(typeless)),
		schema(mv(schema))
	{}
	
	DynamicValuePipeline get(capnp::Field field);
	DynamicValuePipeline get(kj::StringPtr fieldName);
};

}