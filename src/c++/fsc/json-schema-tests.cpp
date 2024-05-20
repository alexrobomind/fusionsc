#include <catch2/catch_test_macros.hpp>

#include "structio-yaml.h"
#include "json-schema.h"

#include <fsc/magnetics.capnp.h>

namespace fsc {

TEST_CASE("field-schema") {
	capnp::Type t = capnp::Type::from<MagneticField>();
	
	/*YAML::Emitter e;
	writeJsonSchema(t, *structio::createVisitor(e));
	
	KJ_DBG(e.c_str());*/
	kj::VectorOutputStream os;
	writeJsonSchema(t, *createVisitor(os, structio::Dialect::JSON));
	
	KJ_DBG(os.getArray().asChars());
}

}