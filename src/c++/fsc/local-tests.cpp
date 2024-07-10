#include <catch2/catch_test_macros.hpp>

#include <kj/thread.h>
#include <kj/string.h>

#include "local.h"

using namespace fsc;

TEST_CASE("local-setup") {
	Library lib = newLibrary();
	
	SECTION("noop") {
	}
	
	SECTION("addref", "Can add refs") {
		Library lib2 = lib -> addRef();
		SUCCEED();
	}
	
	SECTION("th", "Can have a thread handle") {
		ThreadContext ctx(lib->addRef());
		SUCCEED();
		
		SECTION("randworks") {
			kj::FixedArray<byte, 128> a1;
			ctx.rng().randomize(a1);
			
			SUCCEED();
		}
	}
	
	SECTION("no2th", "Can not have more than one thread handle") {
		ThreadContext ctx(lib->addRef());
		
		REQUIRE_THROWS(ThreadContext(lib->addRef()));
	}
}

TEST_CASE("threading") {
	Library lib = newLibrary();
	
	auto h1 = lib->newThread();
	
	SECTION("launchout", "Can open a second handle in another thread") {
		{
			kj::Thread t([&]() {
				ThreadContext h2(lib->addRef());
			});
		}
		SUCCEED();
	}
}

/*
TEST_CASE("oopsie") {
	using namespace kj;
	
	class LambdaProxy {
	public:
		LambdaProxy() {
			KJ_LOG(WARNING, "Lambda constructed");
		}
		
		LambdaProxy(const LambdaProxy& o) {
			KJ_LOG(WARNING, "Lambda copy constructed");
		}
		
		LambdaProxy(LambdaProxy&& o) {
			KJ_LOG(WARNING, "Lambda move constructed");
		}
		
		~LambdaProxy() {
			KJ_LOG(WARNING, "Lambda deleted");
		}
		
		void operator()() {
			KJ_LOG(WARNING, "Lambda called");
		}
	};
	
	auto make_invalid_promise = [] () {
		LambdaProxy l;
		Promise<void> rn = kj::READY_NOW;
		
		return rn.then(l);
	};
	
	EventLoop evl;
	WaitScope ws(evl);
	
	Promise<void> p = make_invalid_promise();
	p.wait(ws);
}
*/
