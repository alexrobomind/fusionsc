#include "data.h"
#include <yaml-cpp/yaml.h>

namespace fsc {
	void load(capnp::DynamicStruct::Builder dst, YAML::Node src);
	
	YAML::Emitter& operator<<(YAML::Emitter&, capnp::DynamicStruct::Reader);
}