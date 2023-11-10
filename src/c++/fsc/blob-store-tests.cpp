#include <catch2/catch_test_macros.hpp>

#include "blob-store.h"
#include "local.h"
#include "sqlite.h"

using namespace fsc;
	
TEST_CASE("blob-store") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto conn = connectSqlite(":memory:");
	
	auto store = createBlobStore(*conn, "blobs");
	
	auto data1 = kj::heapArray<byte>(1024);
	th -> rng().randomize(data1);
	
	auto builder = store -> create(128);
	builder -> write(data1.begin(), data1.size());
	
	auto blob = builder -> finish();
	REQUIRE_THROWS(builder -> finish());
		
	auto data2 = kj::heapArray<byte>(1024);
	{
		auto reader = blob -> open();
		reader -> read(data2.begin(), data2.size());
		REQUIRE(data1 == data2);
	}
	
	{
		auto reader = blob -> open();
		reader -> read(data2.begin(), 512);
		reader -> read(data2.begin() + 512, 512);
		REQUIRE(reader -> read(data2.begin(), 0, 512) == 0);
		REQUIRE(data1 == data2);
	}
	
	REQUIRE(store -> find(blob -> getHash()) != nullptr);
	
	auto builder2 = store -> create(128);
	builder2 -> write(data1.begin(), 213);
	builder2 -> write(data1.begin() + 213, 1024 - 213);
	auto blob2 = builder2 -> finish();
	
	// Check that hashes are cached
	REQUIRE(blob2 -> getId() == blob -> getId());
	
	REQUIRE(blob -> getRefcount() == 2);
	blob -> incRef();
	REQUIRE(blob -> getRefcount() == 3);
	blob -> decRef();
	REQUIRE(blob -> getRefcount() == 2);
	blob -> decRef();
	REQUIRE(blob -> getRefcount() == 1);
	blob -> decRef();
	REQUIRE_THROWS([&]() {
		blob -> getRefcount();
	}());
	
	REQUIRE_THROWS([&]() {
		blob -> open() -> read(data2.begin(), 1);
	}());
}