#include "json.h"
#include "data.h"

#include <catch2/catch_test_macros.hpp>

#include <fsc/magnetics-test.capnp.h>

using namespace fsc;

TEST_CASE("json") {
	auto field = WIRE_FIELD.get();
	
	kj::VectorOutputStream vos;
	writeJson(field, vos);
		
	auto flat = vos.getArray();
	KJ_DBG(flat.asChars());
	
	kj::ArrayInputStream ais(flat);
	Temporary<MagneticField> reread;
	loadJson(reread, ais);
	
	KJ_DBG(field, reread.asReader());
	//KJ_DBG(kj::heapString(arr));
}

TEST_CASE("cbor") {
	auto field = WIRE_FIELD.get();
	
	kj::VectorOutputStream vos;
	writeCbor(field, vos);
		
	auto flat = vos.getArray();
	
	kj::StringTree result;
	for(uint8_t c : flat) {
		auto asStr = kj::hex(c);
		
		if(asStr.size() == 2)
			result = kj::strTree(mv(result), " ", asStr);
		else
			result = kj::strTree(mv(result), " 0", asStr);
	}
	KJ_DBG(result);
	
	kj::ArrayInputStream ais(flat);
	Temporary<MagneticField> reread;
	loadCbor(reread, ais);
	
	KJ_DBG(field, reread.asReader());
	//KJ_DBG(kj::heapString(arr));
}