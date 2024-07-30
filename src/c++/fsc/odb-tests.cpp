#include <catch2/catch_test_macros.hpp>

#include <fsc/data-test.capnp.h>

#include "odb.h"
#include "local.h"
#include "data.h"
#include "sqlite.h"

using namespace fsc;

Warehouse::Folder::Client testWarehouse() {
	auto wh = openWarehouse(*connectSqlite("warehouse"));
	return wh.getRootRequest().send().getRoot();
}

TEST_CASE("warehouse-stress", "[warehouse][.]") {
	constexpr size_t NUM_THREADS = 16;
	constexpr size_t NUM_OBJECTS = 100;
	
	StartupParameters opts;
	opts.numWorkerThreads = 2 * NUM_THREADS;
	
	Library l = newLibrary();
	ThreadContext ctx(l -> addRef());
	
	auto promiseBuilder = kj::heapArrayBuilder<Promise<void>>(NUM_THREADS);
	for(auto i : kj::range(0, NUM_THREADS)) {
		promiseBuilder.add(
			getActiveThread().worker().executeAsync(
				[NUM_OBJECTS, i, conn = connectSqlite("stress-test.sqlite")]() mutable {
					return kj::startFiber(2 * 1024 * 1024, [NUM_OBJECTS, i, conn = mv(conn)](kj::WaitScope& ws) mutable {
						auto wh = openWarehouse(*conn);
						auto root = wh.getRootRequest().send().getRoot();
						
						auto data = kj::heapArray<kj::byte>(128);
						getActiveThread().rng().randomize(data);
						
						auto obj = getActiveThread().dataService().publish(capnp::Data::Reader(data));
						
						for(auto iteration : kj::range(0, NUM_OBJECTS)) {
							auto req = root.putRequest();
							req.setPath(kj::str("object", i));
							req.setValue(obj);
							
							auto storedObj = req.send().wait(ws);
							
							auto downloaded = getActiveThread().dataService().download((DataRef<>::Client) storedObj.getAsGeneric()).wait(ws);
							KJ_REQUIRE(downloaded.getRaw() == data);
							
							getActiveThread().timer().afterDelay(1 * kj::MILLISECONDS).wait(ws);
						}
						
						KJ_DBG("Thread complete", i);
					});
				}
			)
		);
	}
	
	kj::joinPromisesFailFast(promiseBuilder.finish()).wait(ctx.waitScope());
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
	
	auto putResponse = putRequest.send();
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
				KJ_DBG(putResponse.wait(ws));
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
	
	auto egReq = dbRoot.exportGraphRequest();
	egReq.setPath("obj");
	
	auto response = egReq.send().wait(ws);
	KJ_DBG(getActiveThread().dataService().download(response.getGraph()).wait(ws).get());
	
	auto importRequest = dbRoot.importGraphRequest();
	importRequest.setPath("obj");
	importRequest.setGraph(response.getGraph());
	
	auto importResponse = importRequest.send().wait(ws);
	KJ_DBG(importResponse);
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
	
	auto putResponse = putRequest.send();
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
		auto innerCopy = th -> dataService().download(outerCopy.get().getRef()).wait(ws);
		REQUIRE(innerCopy.get() == data);
	}
	
	auto egReq = dbRoot.exportGraphRequest();
	egReq.setPath("obj");
	
	auto response = egReq.send().wait(ws);
	KJ_DBG(getActiveThread().dataService().download(response.getGraph()).wait(ws).get());
	
	// Delete object, then reimport 
	KJ_DBG("Deleting");
	auto rmreq = dbRoot.rmRequest();
	rmreq.setPath("obj");
	rmreq.send().wait(ws);
	KJ_DBG("Deletion done");
	
	auto importRequest = dbRoot.importGraphRequest();
	importRequest.setPath("obj");
	importRequest.setGraph(response.getGraph());
	
	auto importResponse = importRequest.send().wait(ws);
	KJ_DBG(importResponse);
	storedObject = putResponse.getAsGeneric().castAs<HolderRef>();
	
	auto outerCopy2 = th -> dataService().download(storedObject).wait(ws);
	REQUIRE(outerCopy2.get().hasRef() == refHolder.hasRef());
	
	if(refHolder.hasRef()) {
		auto innerCopy = th -> dataService().download(outerCopy2.get().getRef()).wait(ws);
		REQUIRE(innerCopy.get() == data);
	}
	
	auto dcr = dbRoot.deepCopyRequest();
	dcr.setSrcPath("obj");
	dcr.setDstPath("obj2");
	dcr.send().wait(ws);
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