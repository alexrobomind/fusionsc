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
	
	SECTION("raw") {
		auto data = kj::heapArray<const byte>({0, 1, 2, 3, 4});
		
		KJ_LOG(WARNING, "Publishing");
		LocalDataRef<capnp::Data> ref = ds.publish(id, kj::heapArray<const byte>(data));
		
		KJ_LOG(WARNING, "Getting");
		Array<const byte> data2 = ref.getRaw();
		
		REQUIRE(data == data2);
	}
	
	SECTION("anyStruct") {
		capnp::MallocMessageBuilder mb;
		capnp::AnyStruct::Builder sb = mb.getRoot<capnp::AnyPointer>().initAsAnyStruct(10, 0);
		
		auto dataSection = sb.getDataSection();
		for(unsigned int i = 0; i < dataSection.size(); ++i)
			dataSection[i] = i;
		
		LocalDataRef<capnp::AnyStruct> ref = ds.publish<capnp::AnyStruct>(id, sb.asReader());
		Own<capnp::AnyStruct::Reader> reader = ref.get();
		
		REQUIRE(reader->getDataSection() == sb.getDataSection());
	}
}