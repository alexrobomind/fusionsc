#include <catch2/catch_test_macros.hpp>

#include "textio-yaml.h"
#include "json-schema.h"

#include <fsc/magnetics.capnp.h>

namespace fsc {

TEST_CASE("field-schema") {
	capnp::Type t = capnp::Type::from<MagneticField>();
	
	/*YAML::Emitter e;
	writeJsonSchema(t, *textio::createVisitor(e));
	
	KJ_DBG(e.c_str());*/
	kj::VectorOutputStream os;
	writeJsonSchema(t, *createVisitor(os, textio::Dialect::JSON));
	
	KJ_DBG(os.getArray().asChars());
}

}