#include <catch2/catch_test_macros.hpp>

#include "odb.h"
#include "local.h"

using namespace fsc;
using namespace fsc::odb;
	
TEST_CASE("ODB blobstore") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto conn = openSQLite3(":memory:");
	auto t = conn -> beginTransaction();
	
	auto store = kj::refcounted<BlobStore>(*conn, "blobs");
	
	auto data1 = kj::heapArray<byte>(1024);
	th -> rng().randomize(data1);
	
	auto builder = store -> create(128);
	builder.write(data1);
	
	Blob blob = builder.finish();
	
	auto data2 = kj::heapArray<byte>(1024);
	{
		auto reader = blob.open();
		KJ_REQUIRE(reader.read(data2));
		KJ_REQUIRE(reader.remainingOut() == 0);
	}
	
	KJ_IF_MAYBE(pResult, store -> find(blob.hash())) {
	} else {
		KJ_FAIL_REQUIRE("Blob hash not stored");
	}
		
	
	KJ_REQUIRE(data1 == data2);
}

TEST_CASE("ODB open") {
	SECTION("temporary") {
		openObjectDB("");
	}
	
	SECTION("memory") {
		openObjectDB(":memory:");
	}
	
	SECTION("testDB") {
		openObjectDB("testDB");
	}
}