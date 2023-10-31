#include <catch2/catch_test_macros.hpp>

#include <fsc/data-test.capnp.h>

#include "odb.h"
#include "local.h"
#include "data.h"
#include "sqlite.h"

using namespace fsc;
using namespace fsc;

Warehouse::Folder::Client testWarehouse() {
	auto wh = openWarehouse(*connectSqlite("warehouse"));
	return wh.getRootRequest().send().getRoot();
}

TEST_CASE("warehouse-open", "[warehouse]") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	SECTION("temporary") {
		REQUIRE_THROWS(openWarehouse(*connectSqlite("")));
	}
	
	SECTION("memory") {
		REQUIRE_THROWS(openWarehouse(*connectSqlite(":memory:")));
	}
	
	SECTION("testDB") {
		openWarehouse(*connectSqlite("warehouse"));
	}
}

TEST_CASE("warehouse-rw-1", "[warehouse]") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto& ws = th -> waitScope();
	
	using HolderRef = DataRef<test::DataRefHolder<capnp::Data>>;
	
	auto paf = kj::newPromiseAndFulfiller<HolderRef::Client>();
	HolderRef::Client promiseRef(mv(paf.promise));
	
	auto dbRoot = testWarehouse();
	
	auto putRequest = dbRoot.putRequest();
	putRequest.setPath("obj");
	putRequest.setValue(promiseRef.asGeneric());
	
	auto putResponse = putRequest.send().wait(ws);
	auto storedObject = putResponse.getAsGeneric().castAs<HolderRef>();
	
	SECTION("fast termination") {
		// Checks for memory leaks in case download process gets into limbo
	}
	
	SECTION("failure") {
		Maybe<kj::Exception> exc;
		
		SECTION("failure") {	
			exc = KJ_EXCEPTION(FAILED, "TEST FAILED");
		}
		SECTION("disconnected") {
			exc = KJ_EXCEPTION(DISCONNECTED, "TEST DISCONNECTED");
		}
		SECTION("overloaded") {
			exc = KJ_EXCEPTION(OVERLOADED, "TEST OVERLOADED");
		}
		SECTION("unimplemented") {
			exc = KJ_EXCEPTION(UNIMPLEMENTED, "TEST UNIMPLEMENTED");
		}
		
		KJ_IF_MAYBE(pExc, exc) {
			paf.fulfiller -> reject(cp(*pExc));
		
			try {
				storedObject.whenResolved().wait(ws);			
				FAIL("We should never get here");
			} catch(kj::Exception& e) {
				REQUIRE(e.getDescription() == pExc -> getDescription());
				REQUIRE(e.getType() == pExc -> getType());
			}
		} else {
			FAIL("Section without exception");
		}
	}
	
	SECTION("ref") {
		Temporary<test::DataRefHolder<capnp::Data>> refHolder;
		
		auto data = kj::heapArray<byte>(12);
		th -> rng().randomize(data);
		
		SECTION("nestedRefs") {
			refHolder.setRef(th -> dataService().publish(data));
		}
		SECTION("null") {
		}
		
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

TEST_CASE("warehouse-rw-2", "[warehouse]") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	auto& ws = th -> waitScope();
	
	using HolderRef = DataRef<test::DataRefHolder<capnp::Data>>;
	
	auto paf = kj::newPromiseAndFulfiller<HolderRef::Client>();
	HolderRef::Client promiseRef(mv(paf.promise));
	
	auto dbRoot = testWarehouse();
	
	auto putRequest = dbRoot.putRequest();
	putRequest.setPath("obj");
	putRequest.setValue(promiseRef);
	
	auto putResponse = putRequest.send().wait(ws);
	auto storedObject = putResponse.getAsGeneric().castAs<HolderRef>();
	
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
	rmreq.setPath("obj");
	rmreq.send().wait(ws);
}

TEST_CASE("warehouse-rw-3", "[warehouse]") {
	auto data = kj::heapArray<byte>(12);
	
	using HolderRef = DataRef<capnp::Data>;
	
	{
		Library l = newLibrary();
		LibraryThread th = l -> newThread();
		auto& ws = th -> waitScope();	
		
		th -> rng().randomize(data);	
	
		auto dbRoot = testWarehouse();
		
		auto putRequest = dbRoot.putRequest();
		putRequest.setPath("obj");
		putRequest.setValue(th -> dataService().publish(data).asGeneric());
		auto putResponse = putRequest.send().wait(ws);
		
		auto storedObject = putResponse.getAsGeneric();
		storedObject.whenResolved().wait(ws);
	}
	
	
	{
		Library l = newLibrary();
		LibraryThread th = l -> newThread();
		auto& ws = th -> waitScope();		
	
		auto dbRoot = testWarehouse();
		
		auto getRequest = dbRoot.getRequest();
		getRequest.setPath("obj");
		
		auto remoteData = getRequest.send().getAsGeneric().castAs<HolderRef>();
		auto localData = th -> dataService().download(remoteData).wait(ws);
		
		REQUIRE(localData.get() == data);
	}
}