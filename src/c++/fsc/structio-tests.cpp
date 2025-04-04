#include "structio.h"
#include "data.h"

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include <fsc/magnetics-test.capnp.h>

using namespace fsc;
using namespace fsc::structio;

TEST_CASE("structio-r") {
	Temporary<MagneticField> field = WIRE_FIELD.get();
	field.getFilamentField().getFilament().getInline().getData().set(
		0, std::numeric_limits<double>::infinity()
	);
	
	auto vis = createDebugVisitor();
	save(field.asReader(), *vis);
}

TEST_CASE("structio-yaml-anchor") {
	kj::StringPtr input = R"(sum:
- &e1
  invert: {sum : []}
- &e2
  sum: []
- *e1
- <<: *e1
- <<:
  - *e1
- sum: []
)";
		
	Temporary<MagneticField> field;
	load(input.asBytes(), *createVisitor(field), Dialect::YAML);
	KJ_DBG(field);
}

TEST_CASE("structio-rw") {
	KJ_DBG("");
	KJ_DBG("structio PHASE");
	KJ_DBG("");
	
	auto l = newLibrary();
	auto lt = l -> newThread();
	auto& ws = lt -> waitScope();
	
	Temporary<MagneticField> nestedField;
	{
		auto sum = nestedField.initSum(2);
		sum[0].setRef(nullptr);
		sum[1].initNested();
	}
	
	Temporary<MagneticField> field;
	{
		auto sum = field.initSum(4);
		sum.setWithCaveats(2, WIRE_FIELD.get());
		sum[2].getFilamentField().getFilament().getInline().getData().set(
			0, std::numeric_limits<double>::infinity()
		);
		sum[3].setRef(lt -> dataService().publish(nestedField.asReader()));
	}
	
	Dialect opts(Dialect::JSON);
	SaveOptions sOpts;
	SECTION("json") {
		opts.language = Dialect::JSON;
		
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
		
		SECTION("compact") {
			sOpts.compact = true;
		}
	}
	SECTION("bson") {
		opts.language = Dialect::BSON;
	}
	SECTION("cbor") {
		opts.language = Dialect::CBOR;
		
		SECTION("ikeys") {
			sOpts.integerKeys = true;
		}
		SECTION("compact") {
			sOpts.compact = true;
		}
		SECTION("icompact") {
			sOpts.integerKeys = true;
			sOpts.compact = true;
		}
		SECTION("usualkeys") {
		}
	}
	SECTION("yaml") {
		opts.language = Dialect::YAML;
		
		SECTION("ikeys") {
			sOpts.integerKeys = true;
		}
		SECTION("compact") {
			sOpts.compact = true;
		}
		SECTION("icompact") {
			sOpts.integerKeys = true;
			sOpts.compact = true;
		}
		SECTION("usualkeys") {
		}
	}
	
	KJ_DBG(opts.language);
	
	kj::VectorOutputStream vos;
	save(field.asReader(), vos, opts, sOpts, ws);
		
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
	
	Temporary<MagneticField> reread;
	// load(flat, *createDebugVisitor(), opts);
	load(flat, *createVisitor(reread), opts);
	
	KJ_DBG(field);
	KJ_DBG(reread.asReader());
}