#include <catch2/catch_test_macros.hpp>

#include <kj/thread.h>
#include <kj/string.h>

#include "data.h"

using namespace fsc;

TEST_CASE("local_publish") {
	Library l = newLibrary();
	
	LocalDataService ds(l);
	
	auto id   = kj::heapArray<const byte>({0x00, 0xFF});
	auto data = kj::heapArray<const byte>({0, 1, 2, 3, 4});
	
	LocalDataRef<capnp::Data> ref = ds.publish(id, mv(data));
	Array<const byte> data2 = ref.getRaw();
	
	REQUIRE(data == data2);
}
