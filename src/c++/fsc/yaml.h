#include "data.h"
#include <yaml-cpp/yaml.h>

namespace fsc {
	void load(capnp::DynamicStruct::Builder dst, YAML::Node src);
	void load(capnp::DynamicList::Builder dst, YAML::Node src);
	
	capnp::DynamicValue::Reader loadPrimitive(capnp::Type type, YAML::Node src);
	
	YAML::Emitter& operator<<(YAML::Emitter&, capnp::DynamicStruct::Reader);
	YAML::Emitter& operator<<(YAML::Emitter&, capnp::DynamicList::Reader);
	YAML::Emitter& operator<<(YAML::Emitter&, capnp::DynamicValue::Reader);
}