#include "jtext.h"
#include "../http.h"

#include <fsc/devices/w7x-test.capnp.h>

#include <catch2/catch_test_macros.hpp>

namespace fsc {
	
namespace devices { namespace jtext {

TEST_CASE("jtext-geo-resolver") {
	auto l = newLibrary();
	auto lt = l -> newThread();
	auto& ws = lt -> waitScope();
	
	auto gs = newGeometryResolver();
	auto resolveRequest = gs.resolveGeometryRequest();
	auto geo = resolveRequest.initGeometry();
	
	SECTION("target") {
		geo.initJtext().setTarget();
	}
	
	SECTION("lfsLimiter") {
		geo.initJtext().setLfsLimiter(0.235);
	}
	
	auto response = resolveRequest.send().wait(ws);	
}

TEST_CASE("jtext-field-resolver") {
	auto l = newLibrary();
	auto lt = l -> newThread();
	auto& ws = lt -> waitScope();
	
	auto gs = newFieldResolver();
	auto resolveRequest = gs.resolveFieldRequest();
	auto field = resolveRequest.initField();
	
	SECTION("filament") {
		auto fil = field.initFilamentField().getFilament();
		fil.initJtext().setIslandCoil(1);
	}
	
	auto response = resolveRequest.send().wait(ws);	
}

}}

}