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
		ArrayPtr<const byte> data2 = ref.getRaw();
		
		REQUIRE(data == data2);
	}
	
	SECTION("anyStruct") {
		capnp::MallocMessageBuilder mb;
		capnp::AnyStruct::Builder sb = mb.getRoot<capnp::AnyPointer>().initAsAnyStruct(10, 0);
		
		auto dataSection = sb.getDataSection();
		for(unsigned int i = 0; i < dataSection.size(); ++i)
			dataSection[i] = i;
		
		LocalDataRef<capnp::AnyStruct> ref = ds.publish<capnp::AnyStruct>(id, sb.asReader());
		capnp::AnyStruct::Reader reader = ref.get();
		
		REQUIRE(reader.getDataSection() == sb.getDataSection());
	}

	SECTION("testData") {
		using fsc::test::DataHolder;
		using fsc::test::DataRefHolder;
		
		using DDH = DataRefHolder<DataHolder>;

		auto& ws = th -> ioContext().waitScope;
		auto data1 = kj::heapArray<const byte>({0x00, 0x01});

		capnp::MallocMessageBuilder mb1;
		DataHolder::Builder inner = mb1.initRoot<DataHolder>();
		inner.setData(data1);
		LocalDataRef<DataHolder> ref1 = ds.publish<DataHolder>({0x00}, inner.asReader());

		capnp::MallocMessageBuilder mb2;
		DDH::Builder refHolder = mb2.initRoot<DDH>();
		refHolder.setRef(ref1);
		LocalDataRef<DDH> ref2 = ds.publish<DDH>({0x01}, refHolder.asReader());

		DDH::Reader refHolder2 = ref2.get();
		LocalDataRef<DataHolder> ref12 = ds.download<DataHolder>(refHolder2.getRef()).wait(ws);
		DataHolder::Reader inner2 = ref12.get();

		REQUIRE(inner.getData() == inner2.getData());
	}
}

TEST_CASE("remote_publish") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	KJ_LOG(WARNING, "Starting data service");
	LocalDataService ds1(l);
	
	Library l2 = newLibrary();
	KJ_LOG(WARNING, "Starting second service");
	LocalDataService ds2(l);
	
	SECTION("testData") {
		using fsc::test::DataHolder;
		using fsc::test::DataRefHolder;
		
		using DDH = DataRefHolder<DataHolder>;

		auto& ws = th -> ioContext().waitScope;
		auto data1 = kj::heapArray<const byte>({0x00, 0x01});

		capnp::MallocMessageBuilder mb1;
		DataHolder::Builder inner = mb1.initRoot<DataHolder>();
		inner.setData(data1);
		LocalDataRef<DataHolder> ref1 = ds1.publish<DataHolder>({0x0}, inner.asReader());

		capnp::MallocMessageBuilder mb2;
		DDH::Builder refHolder = mb2.initRoot<DDH>();
		refHolder.setRef(ref1);
		LocalDataRef<DDH> ref2 = ds1.publish<DDH>({0x1}, refHolder.asReader());

		LocalDataRef<DDH> ref22 = ds2.download<DDH>(ref2).wait(ws);
		DDH::Reader refHolder2 = ref22.get();
		LocalDataRef<DataHolder> ref12 = ds2.download<DataHolder>(refHolder2.getRef()).wait(ws);
		DataHolder::Reader inner2 = ref12.get();

		REQUIRE(inner.getData() == inner2.getData());
	}
}