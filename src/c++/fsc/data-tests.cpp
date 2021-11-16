#include <catch2/catch_test_macros.hpp>

#include <kj/thread.h>
#include <kj/string.h>
#include <kj/exception.h>

#include <fsc/data-test.capnp.h>

#include "data.h"

using namespace fsc;

TEST_CASE("local_publish") {
	//kj::printStackTraceOnCrash();
	
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
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

	SECTION("testData") {
		using fsc::test::DataHolder;
		using fsc::test::DataRefHolder;
		
		using DDH = DataRefHolder<DataHolder>;

		auto& ws = th -> ioContext().waitScope;

		KJ_LOG(WARNING, "Building message 1");
		capnp::MallocMessageBuilder mb1;
		auto data1 = kj::heapArray<const byte>({0x00, 0x01});
		DataHolder::Builder inner = mb1.initRoot<DataHolder>();
		KJ_LOG(WARNING, "  Setting data");
		inner.setData(data1);
		KJ_LOG(WARNING, "  Publishing");
		LocalDataRef<DataHolder> ref1 = ds.publish<DataHolder>({0x00}, inner.asReader());

		KJ_LOG(WARNING, "Building message 2");
		capnp::MallocMessageBuilder mb2;
		DDH::Builder refHolder = mb2.initRoot<DDH>();
		refHolder.setRef(ref1);
		LocalDataRef<DDH> ref2 = ds.publish<DDH>({0x01}, refHolder.asReader());

		KJ_LOG(WARNING, "Downloading ref");
		Own<DDH::Reader> refHolder2 = ref2.get();
		LocalDataRef<DataHolder> ref12 = ds.download<DataHolder>(refHolder2->getRef()).wait(ws);
		Own<DataHolder::Reader> inner2 = ref12.get();

		REQUIRE(inner.getData() == inner2 -> getData());
	}
}
