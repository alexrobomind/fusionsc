#include <catch2/catch_test_macros.hpp>

#include <kj/thread.h>
#include <kj/string.h>
#include <kj/exception.h>

#include "data.h"

using namespace fsc;

TEST_CASE("local_publish") {
	kj::printStackTraceOnCrash();
	
	Library l = newLibrary();
	
	KJ_LOG(WARNING, "Starting data service");
	LocalDataService ds(l);
	
	auto id   = kj::heapArray<const byte>({0x00, 0xFF});
	auto data = kj::heapArray<const byte>({0, 1, 2, 3, 4});
	
	KJ_LOG(WARNING, "Publishing");
	LocalDataRef<capnp::Data> ref = ds.publish(id, mv(data));
	
	KJ_LOG(WARNING, "Getting");
	Array<const byte> data2 = ref.getRaw();
	
	REQUIRE(data == data2);
}
