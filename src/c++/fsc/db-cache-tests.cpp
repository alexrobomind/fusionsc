#include <catch2/catch_test_macros.hpp>

#include "db-cache.h"

TEST_CASE("db-cache") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto& ws = th -> waitScope();
	
	auto array = kj::heapArray<kj::byte>(12);
	th -> rng().randomize(array);
	
	auto ref = th -> dataService().publish<capnp::Data>(array);
	
	auto cache = createCache();
	auto ref2 = th -> dataService().download(cache -> cache(ref)).wait(ws);
	
	REQUIRE(ref.get() == ref2.get());
}