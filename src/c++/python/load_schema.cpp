#include "fscpy.h"
#include "async.h"
#include "loader.h"

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

namespace fscpy {

void loadDefaultSchema(py::module_& m) {
	auto schemas = getBuiltinSchemas<DataService>();
	
	for(auto node : schemas) {
		defaultLoader.add(node);
	}
	
	for(auto node : schemas) {
		defaultLoader.importNodeIfRoot(node.getId(), m);
	}
}

}