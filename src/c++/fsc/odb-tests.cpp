#include <catch2/catch_test_macros.hpp>

#include "odb.h"
#include "local.h"

namespace fsc {
	
TEST_CASE("ODB write-read") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	KJ_DBG("Opening DB");
	auto conn = openSQLite3(":memory:");
	KJ_DBG("Connected");
	
	auto store = kj::refcounted<BlobStore>(*conn, "blobs");
	KJ_DBG("Store created");
	
	auto data1 = kj::heapArray<byte>(1024);
	th -> rng().randomize(data1);
	
	BlobBuilder builder(*store);
	KJ_DBG("Builder created");
	builder.write(data1);
	KJ_DBG("Data written");
	
	Blob blob = builder.finish();
	
	auto data2 = kj::heapArray<byte>(1024);
	{
		auto reader = blob.open();
		KJ_REQUIRE(reader.read(data2));
		KJ_REQUIRE(reader.remainingOut() == 0);
	}
	
	KJ_REQUIRE(data1 == data2);
}

}