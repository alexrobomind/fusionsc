#include <catch2/catch_test_macros.hpp>

#include <kj/thread.h>
#include <kj/string.h>
#include <kj/exception.h>
#include <kj/filesystem.h>

#include <fsc/data-test.capnp.h>
#include <fsc/data-test.capnp.cu.h>

#include <capnp/serialize-text.h>

#include "data.h"

using namespace fsc;

TEST_CASE("local_publish") {
	//kj::printStackTraceOnCrash();
	
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	LocalDataService ds(l);
	ds.setChunkDebugMode();
	
	auto id   = kj::heapArray<const byte>({0x00, 0xFF});
	
	SECTION("raw") {
		auto data = kj::heapArray<const byte>({0, 1, 2, 3, 4});
		
		LocalDataRef<capnp::Data> ref = ds.publish(kj::heapArray<const byte>(data));
		ArrayPtr<const byte> data2 = ref.getRaw();
		
		REQUIRE(data == data2);
	}
	
	SECTION("anyStruct") {
		capnp::MallocMessageBuilder mb;
		capnp::AnyStruct::Builder sb = mb.getRoot<capnp::AnyPointer>().initAsAnyStruct(10, 0);
		
		auto dataSection = sb.getDataSection();
		for(unsigned int i = 0; i < dataSection.size(); ++i)
			dataSection[i] = i;
		
		LocalDataRef<capnp::AnyStruct> ref = ds.publish(sb);
		capnp::AnyStruct::Reader reader = ref.get();
		
		REQUIRE(reader.getDataSection() == sb.getDataSection());
	
		/*SECTION("capability") {
			capnp::MallocMessageBuilder mb2;
			auto id2  = kj::heapArray<const byte>({0x01, 0xFF});
			capnp::AnyStruct::Builder sb2 = mb.getRoot<capnp::AnyPointer>().initAsAnyStruct(0, 1);
			sb2.getPointerSection().set(0, ref);
			
			LocalDataRef<capnp::AnyStruct> ref2 = ds.publish<capnp::AnyStruct>(id2, sb.asReader());
			Own<capnp::AnyStruct::Reader> reader2 = ref.get();
			
			LocalDataRef<capnp::AnyStruct> ref3 = ds.download()
		}*/
	}

	SECTION("dataTransfer") {
		using fsc::test::DataHolder;
		using fsc::test::DataRefHolder;
		
		using DDH = DataRefHolder<DataHolder>;

		auto& ws = th -> ioContext().waitScope;
		auto data1 = kj::heapArray<byte>(1024);
		th -> rng().randomize(data1);

		capnp::MallocMessageBuilder mb1;
		DataHolder::Builder inner = mb1.initRoot<DataHolder>();
		inner.setData(data1);
		LocalDataRef<DataHolder> ref1 = ds.publish(inner);

		capnp::MallocMessageBuilder mb2;
		DDH::Builder refHolder = mb2.initRoot<DDH>();
		refHolder.setRef(ref1);
		LocalDataRef<DDH> ref2 = ds.publish(refHolder);
		
		SECTION("local") {
			LocalDataRef<DataHolder> ref12 = ds.download(ref2.get().getRef()).wait(ws);
			DataHolder::Reader inner2 = ref12.get();
			REQUIRE(inner.getData() == inner2.getData());
		}
		
		SECTION("local-recursive") {
			LocalDataRef<DataHolder> ref12 = ds.download(ref2.get().getRef(), true).wait(ws);
			DataHolder::Reader inner2 = ref12.get();
			REQUIRE(inner.getData() == inner2.getData());
		}
		
		SECTION("local_bypass") {
			LocalDataService ds2(l);
			
			LocalDataRef<DataHolder> ref12 = ds2.download(ref2.get().getRef()).wait(ws);
			DataHolder::Reader inner2 = ref12.get();
			REQUIRE(inner.getData() == inner2.getData());
		}
		
		SECTION("remote") {
			Library l2 = newLibrary();
			LocalDataService ds2(l2);
			
			LocalDataRef<DataHolder> ref12 = ds2.download(
				ds2.download(ref2).wait(ws).get().getRef()
			).wait(ws);
			DataHolder::Reader inner2 = ref12.get();
			REQUIRE(inner.getData() == inner2.getData());
		}
		
		/*SECTION("archive") {
			capnp::MallocMessageBuilder tmp;
			auto archive = tmp.initRoot<Archive>();
			
			INFO("Building");
			ds.buildArchive(ref2, archive).wait(ws);
			
			Library l2 = newLibrary();
			LocalDataService ds2(l2);
			
			INFO("Reading");
			LocalDataRef<DDH> ref22 = ds2.publishArchive<DDH>(archive);
			
			INFO("Downloading");
			LocalDataRef<DataHolder> ref12 = ds2.download(ref22.get().getRef()).wait(ws);
			
			INFO("Extracting");
			DataHolder::Reader inner2 = ref12.get();
			REQUIRE(inner.getData() == inner2.getData());
		}*/
		
		SECTION("tmpfile-archive") {
			INFO("opening");
			//Own<const kj::File> file = kj::newInMemoryFile(kj::systemCoarseCalendarClock());
			Own<const kj::File> file = kj::newDiskFilesystem()->getCurrent().createTemporary();
			//Own<const kj::File> file = kj::newDiskFilesystem()->getCurrent().openFile(kj::Path("testFile"), kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
			INFO("checking");
			*file;
			INFO("writing");
			ds.writeArchive(ref2, *file).wait(ws);
			
			Library l2 = newLibrary();
			LocalDataService ds2(l2);
			
			INFO("reading");
			LocalDataRef<DDH> ref22 = ds2.publishArchive<DDH>(*file);
			LocalDataRef<DataHolder> ref12 = ds2.download(ref22.get().getRef()).wait(ws);
			
			DataHolder::Reader inner2 = ref12.get();
			REQUIRE(inner.getData() == inner2.getData());
		}
		
		SECTION("flatrep") {
			auto asFlat = ds.downloadFlat(ref2).wait(ws);
			
			Library l2 = newLibrary();
			LocalDataService ds2(l2);
			
			LocalDataRef<DDH> ref22 = ds2.publishFlat<DDH>(mv(asFlat));
			LocalDataRef<DataHolder> ref12 = ds2.download(ref22.get().getRef()).wait(ws);
			
			DataHolder::Reader inner2 = ref12.get();
			REQUIRE(inner.getData() == inner2.getData());
		}
	}
}

TEST_CASE("check-ordinal") {
	test::TestStruct::Reader tests[] = {
		test::TEST0.get(),
		test::TEST1.get(),
		test::TEST2.get(),
		test::TEST3.get(),
		test::TEST4.get(),
		test::TEST5.get(),
		test::TEST6.get(),
		test::TEST7.get(),
		test::TEST8.get(),
		test::TEST9.get(),
		test::TEST10.get(),
		test::TEST11.get()
	};
	const unsigned int nTests = 12;
	
	for(unsigned int i = 0; i < nTests; ++i) {
		// KJ_LOG(WARNING, "Executing positive test ", i);
		REQUIRE(hasMaximumOrdinal(tests[i], i));
		
		// KJ_LOG(WARNING, "Executing negative test ", i);
		if(i > 1)
			REQUIRE(!hasMaximumOrdinal(tests[i], i - 1));
	}
}