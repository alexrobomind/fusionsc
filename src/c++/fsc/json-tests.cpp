#include "json.h"
#include "data.h"

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include <fsc/magnetics-test.capnp.h>

using namespace fsc;

TEST_CASE("json") {
	Temporary<MagneticField> field = WIRE_FIELD.get();
	field.getFilamentField().getFilament().getInline().getData().set(
		0, std::numeric_limits<double>::infinity()
	);
	
	JsonOptions opts;
	SECTION("json") {
		opts.dialect = JsonOptions::JSON;
		
		SECTION("yaml-style") {
		}
		
		SECTION("py-style") {
			opts.jsonNan = "NaN";
			opts.jsonInf = "Infinity";
			opts.jsonNegInf = "-Infinity";
		}
		
		SECTION("mangled") {
			opts.jsonNan = "Oo";
			opts.jsonInf = "Orgalorg";
			opts.jsonNegInf = "SpecialRelativistics";
		}
	}
	SECTION("bson") {
		opts.dialect = JsonOptions::BSON;
	}
	SECTION("cbor") {
		opts.dialect = JsonOptions::CBOR;
	}
	
	kj::VectorOutputStream vos;
	writeJson(field.asReader(), vos, opts);
		
	auto flat = vos.getArray();
	KJ_DBG(flat.asChars());
	
	kj::StringTree hex;
	for(uint8_t c : flat) {
		auto asStr = kj::hex(c);
		
		if(asStr.size() == 2)
			hex = kj::strTree(mv(hex), " ", asStr);
		else
			hex = kj::strTree(mv(hex), " 0", asStr);
	}
	KJ_DBG(hex);
	
	kj::ArrayInputStream ais(flat);
	Temporary<MagneticField> reread;
	loadJson(reread, ais, opts);
	
	KJ_DBG(field);
	KJ_DBG(reread.asReader());
}