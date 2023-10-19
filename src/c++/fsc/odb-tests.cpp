#include <catch2/catch_test_macros.hpp>

#include <fsc/data-test.capnp.h>

#include "odb.h"
#include "local.h"
#include "data.h"
#include "sqlite.h"

using namespace fsc;
using namespace fsc::odb;

TEST_CASE("ODB open") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	SECTION("temporary") {
		openObjectDb(*connectSqlite(""));
	}
	
	SECTION("memory") {
		openObjectDb(*connectSqlite(":memory:"));
	}
	
	SECTION("testDB") {
		openObjectDb(*connectSqlite("testDB"));
	}
}

TEST_CASE("ODB rw") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto& ws = th -> waitScope();
	
	using HolderRef = DataRef<test::DataRefHolder<capnp::Data>>;
	
	auto paf = kj::newPromiseAndFulfiller<HolderRef::Client>();
	HolderRef::Client promiseRef(mv(paf.promise));
	
	Folder::Client dbRoot = openObjectDb(*connectSqlite(":memory:"));
	
	auto putRequest = dbRoot.putEntryRequest();
	putRequest.setName("obj");
	putRequest.setValue(promiseRef.asGeneric());
	
	auto putResponse = putRequest.send().wait(ws);
	auto storedObject = putResponse.getValue().castAs<HolderRef>();
	
	SECTION("fast termination") {
		// Checks for memory leaks in case download process gets into limbo
	}
	
	SECTION("failure") {
		kj::Exception errors[4] = {
			KJ_EXCEPTION(FAILED),
			KJ_EXCEPTION(DISCONNECTED),
			KJ_EXCEPTION(OVERLOADED),
			KJ_EXCEPTION(UNIMPLEMENTED)
		};
		
		for(auto exc : errors) {
			auto paf = kj::newPromiseAndFulfiller<HolderRef::Client>();
			HolderRef::Client promiseRef(mv(paf.promise));
			
			auto putRequest = dbRoot.putEntryRequest();
			putRequest.setName("obj");
			putRequest.setValue(promiseRef);
			
			auto putResponse = putRequest.send().wait(ws);
			auto storedObject = putResponse.getValue();
			
			auto msg = "Injected failure message"_kj;
			exc.setDescription(str(msg));
			paf.fulfiller -> reject(mv(exc));
			
			try {
				storedObject.whenResolved().wait(ws);			
				FAIL("We should never get here");
			} catch(kj::Exception& e) {
				REQUIRE(e.getDescription() == msg);
				REQUIRE(e.getType() == exc.getType());
			}
		}
	}
	
	SECTION("ref") {
		Temporary<test::DataRefHolder<capnp::Data>> refHolder;
		
		auto data = kj::heapArray<byte>(12);
		th -> rng().randomize(data);
		
		SECTION("nestedRefs") {
			refHolder.setRef(th -> dataService().publish(data));
		}
		SECTION("null") {}
		
		paf.fulfiller -> fulfill(th -> dataService().publish(refHolder.asReader()));
		storedObject.whenResolved().wait(ws);
		
		auto outerCopy = th -> dataService().download(storedObject).wait(ws);
		REQUIRE(outerCopy.get().hasRef() == refHolder.hasRef());
		
		if(refHolder.hasRef()) {
			auto innerCopy = th -> dataService().download(refHolder.getRef()).wait(ws);
			REQUIRE(innerCopy.get() == data);
		}
	}
}

TEST_CASE("ODB rw main") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto& ws = th -> waitScope();
	
	using HolderRef = DataRef<test::DataRefHolder<capnp::Data>>;
	
	auto paf = kj::newPromiseAndFulfiller<HolderRef::Client>();
	HolderRef::Client promiseRef(mv(paf.promise));
	
	Folder::Client dbRoot = openObjectDb(*connectSqlite(":memory:"));
	
	auto putRequest = dbRoot.putEntryRequest();
	putRequest.setName("obj");
	putRequest.setValue(promiseRef);
	
	auto putResponse = putRequest.send().wait(ws);
	auto storedObject = putResponse.getValue().castAs<HolderRef>();
	
	Temporary<test::DataRefHolder<capnp::Data>> refHolder;
	
	auto data = kj::heapArray<byte>(12);
	th -> rng().randomize(data);
	
	SECTION("nestedRefs") {
		refHolder.setRef(th -> dataService().publish(data));
	}
	SECTION("null") {}
	
	paf.fulfiller -> fulfill(th -> dataService().publish(refHolder.asReader()));
	storedObject.whenResolved().wait(ws);
	
	auto outerCopy = th -> dataService().download(storedObject).wait(ws);
	REQUIRE(outerCopy.get().hasRef() == refHolder.hasRef());
	
	if(refHolder.hasRef()) {
		auto innerCopy = th -> dataService().download(refHolder.getRef()).wait(ws);
		REQUIRE(innerCopy.get() == data);
	}
	
	auto rmreq = dbRoot.rmRequest();
	rmreq.setName("obj");
	rmreq.send().wait(ws);
}

TEST_CASE("ODB rw persistent") {
	auto data = kj::heapArray<byte>(12);
	
	using HolderRef = DataRef<capnp::Data>;
	
	{
		Library l = newLibrary();
		LibraryThread th = l -> newThread();
		auto& ws = th -> waitScope();	
		
		th -> rng().randomize(data);	
	
		Folder::Client dbRoot = openObjectDb(*connectSqlite("testdb.sqlite"));
		
		auto putRequest = dbRoot.putEntryRequest();
		putRequest.setName("obj");
		putRequest.setValue(th -> dataService().publish(data).asGeneric());
		auto putResponse = putRequest.send().wait(ws);
		
		auto storedObject = putResponse.getValue().castAs<HolderRef>();
		
		storedObject.whenResolved().wait(ws);
	}
	
	
	{
		Library l = newLibrary();
		LibraryThread th = l -> newThread();
		auto& ws = th -> waitScope();		
	
		Folder::Client dbRoot = openObjectDb(*connectSqlite("testdb.sqlite"));
		
		auto getRequest = dbRoot.getEntryRequest();
		getRequest.setName("obj");
		
		auto getResponse = getRequest.send().wait(ws);
		auto remoteData = getResponse.getValue().castAs<HolderRef>();
		auto localData = th -> dataService().download(remoteData).wait(ws);
		
		REQUIRE(localData.get() == data);
	}
}