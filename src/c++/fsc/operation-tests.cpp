#include "operation.h"

using namespace fsc;

#include <catch2/catch_test_macros.hpp>
	
struct DestroyedIn {
	static inline const kj::Executor* pThread;
	~DestroyedIn() { pThread = &kj::getCurrentThreadExecutor(); }
};

TEST_CASE("operation-promises") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	{
		Operation op1;
		Promise<void> p = op1.whenDone();
		
		op1.done();
		
		REQUIRE(p.poll(ws));
	}
	
	{
		Operation op1;
		Promise<void> p = op1.whenDone();
		
		op1.fail(KJ_EXCEPTION(FAILED, "Test exception"));
		
		REQUIRE(p.poll(ws));
		REQUIRE_THROWS(p.wait(ws));
	}
}

TEST_CASE("operation-lifecycle") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	{
		Operation op1;
		op1.attachDestroyHere(DestroyedIn());
	}
	
	REQUIRE(DestroyedIn::pThread == &kj::getCurrentThreadExecutor());
}