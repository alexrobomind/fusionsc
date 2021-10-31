#include <catch2/catch_test_macros.hpp>

#include <kj/array.h>

#include "random.h"

TEST_CASE("random-checks") {
	fsc::CSPRNG rng;
	
	SECTION("2notsame") {
		auto a1 = kj::heapArray<kj::byte>(128);
		auto a2 = kj::heapArray<kj::byte>(128);
		
		rng.randomize(a1);
		rng.randomize(a2);
		
		REQUIRE(a1 != a2);
	}
}