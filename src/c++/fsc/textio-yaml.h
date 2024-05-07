#pragma once

#include "data.h"
#include "textio.h"
#include <yaml-cpp/yaml.h>

namespace fsc { namespace textio {
	Own<Visitor> createVisitor(YAML::Emitter&);
}}

namespace fsc {
	template<typename T>
	kj::String asYaml(T&&);
}

	
YAML::Emitter& operator<<(YAML::Emitter&, capnp::DynamicValue::Reader);

// Implementation

namespace fsc {
	template<typename T>
	kj::String asYaml(T&& in) {
		YAML::Emitter e;
		e << fwd<T>(in);
		return kj::heapString(e.c_str());
	}
}