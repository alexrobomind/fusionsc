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
		auto op1 = newOperation();
		Promise<void> p = op1->whenDone();
		
		op1->done();
		
		REQUIRE(p.poll(ws));
	}
	
	{
		auto op1 = newOperation();
		Promise<void> p = op1->whenDone();
		
		op1->fail(KJ_EXCEPTION(FAILED, "Test exception"));
		
		REQUIRE(p.poll(ws));
		REQUIRE_THROWS(p.wait(ws));
	}
}

TEST_CASE("operation-lifecycle") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	{
		auto op1 = newOperation();
		op1->attachDestroyHere(DestroyedIn());
	}
	
	REQUIRE(DestroyedIn::pThread == &kj::getCurrentThreadExecutor());
}