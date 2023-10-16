#include <catch2/catch_test_macros.hpp>

#include "db-cache.h"
#include "local.h"
#include "common.h"
#include "data.h"

using namespace fsc;

TEST_CASE("db-cache") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto& ws = th -> waitScope();
	
	auto array = kj::heapArray<kj::byte>(12);
	th -> rng().randomize(array);
	
	auto conn = openSQLite3(":memory:");
	auto bs = createBlobStore(*conn);
	auto cache = createDBCache(*bs);
	
	auto ref = th -> dataService().publish(capnp::Data::Reader(array));
	
	auto ref2 = th -> dataService().download(cache -> cache(ref)).wait(ws);
	
	REQUIRE(ref.get() == ref2.get());
}