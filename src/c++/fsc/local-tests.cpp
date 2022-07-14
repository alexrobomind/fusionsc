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
		ThreadHandle th(lib->addRef());
		SUCCEED();
		
		SECTION("randworks") {
			kj::FixedArray<byte, 128> a1;
			th.rng().randomize(a1);
			
			SUCCEED();
		}
	}
	
	SECTION("no2th", "Can not have more than one thread handle") {
		ThreadHandle th1(lib->addRef());
		
		REQUIRE_THROWS(ThreadHandle(lib->addRef()));
	}
}

TEST_CASE("threading") {
	Library lib = newLibrary();
	
	auto h1 = lib->newThread();
	
	SECTION("launchout", "Can open a second handle in another thread") {
		{
			kj::Thread t([&]() {
				ThreadHandle h2(lib->addRef());
			});
		}
		SUCCEED();
	}
	
	SECTION("selfconnect", "Can self connect") {
		CrossThreadConnection conn;
		
		SECTION("accept_then_connect") {
			auto streamAcc = conn.accept(*h1);
			auto streamConn = conn.connect(*h1);
			
			SECTION("wait_accept_first") {
				streamAcc.wait(h1->waitScope());
				streamConn.wait(h1->waitScope());
				SUCCEED();
			}
			
			SECTION("wait_connect_first") {
				streamConn.wait(h1->waitScope());
				streamAcc.wait(h1->waitScope());
				SUCCEED();
			}
		}
		
		SECTION("connect_then_accept") {
			auto streamConn = conn.connect(*h1);
			auto streamAcc = conn.accept(*h1);
			
			SECTION("wait_accept_first") {
				streamAcc.wait(h1->waitScope());
				streamConn.wait(h1->waitScope());
				SUCCEED();
			}
			
			SECTION("wait_connect_first") {
				streamConn.wait(h1->waitScope());
				streamAcc.wait(h1->waitScope());
				SUCCEED();
			}
		}
	}
	
	SECTION("connect", "Can connect two threads (repeats multiple times)") {
	for(size_t i = 0; i < 10; ++i) { DYNAMIC_SECTION("repeat" << i) {
		CrossThreadConnection conn;
		
		kj::FixedArray<byte, 4> recvBuf;
		kj::FixedArray<byte, 4> sendBuf;
		
		h1->rng().randomize(sendBuf);
		
		SECTION("mainaccepts") {
			kj::Thread t([&lib, &conn, &sendBuf]() {
				auto h2 = lib->newThread();
				auto streamPromise = conn.connect(*h2);
				auto stream = streamPromise.wait(h2->waitScope());
				
				stream -> write(sendBuf.begin(), sendBuf.size()).wait(h2->waitScope());
			});
		
			auto streamPromise = conn.accept(*h1);
			auto stream = streamPromise.wait(h1->waitScope());
			
			stream -> read(recvBuf.begin(), recvBuf.size()).wait(h1->waitScope());
	
			REQUIRE((ArrayPtr<byte>) sendBuf == (ArrayPtr<byte>) recvBuf);
		}
		
		SECTION("mainconnects") {
			kj::Thread t([&lib, &conn, &sendBuf]() {
				auto h2 = lib->newThread();
				auto streamPromise = conn.accept(*h2);
				auto stream = streamPromise.wait(h2->waitScope());
				
				stream -> write(sendBuf.begin(), sendBuf.size()).wait(h2->waitScope());
			});
		
			auto streamPromise = conn.connect(*h1);
			auto stream = streamPromise.wait(h1->waitScope());
			
			stream -> read(recvBuf.begin(), recvBuf.size()).wait(h1->waitScope());
	
			REQUIRE((ArrayPtr<byte>) sendBuf == (ArrayPtr<byte>) recvBuf);
		}
	}}
	}
}

TEST_CASE("pipe") {
	Library lib = newLibrary();
	
	auto h1 = lib->newThread();
	
	auto pipe = newPipe();
	
	constexpr size_t nBytes = 256;
	auto rBuf = kj::heapArray<byte>(nBytes);
	auto wBuf = kj::heapArray<byte>(nBytes);
	
	h1->rng().randomize(rBuf);
	
	SECTION("single-thread") {
		KJ_DBG("Writing");
		auto p1 = pipe.ends[0] -> write(rBuf.begin(), rBuf.size());
		KJ_DBG("Reading");
		auto p2 = pipe.ends[1] -> read(wBuf.begin(), wBuf.size(), wBuf.size());
		
		KJ_DBG("Wait read");
		p2.wait(h1->waitScope());
		KJ_DBG("Wait write");
		p1.wait(h1->waitScope());
	}
	
	SECTION("multithread", "Can open a second handle in another thread") {
		{
			kj::Thread t([&]() {
				ThreadHandle h2(lib->addRef());
				auto p2 = pipe.ends[1] -> read(wBuf.begin(), wBuf.size(), wBuf.size());
				p2.wait(h2.waitScope());
				
				pipe.ends[1] -> abortRead();
			});
			
			auto p1 = pipe.ends[0] -> write(rBuf.begin(), rBuf.size());
			p1.wait(h1->waitScope());
		}
	}
	
	SECTION("multithread-reverse", "Can open a second handle in another thread") {
		{
			kj::Thread t([&]() {
				ThreadHandle h2(lib->addRef());
				auto p2 = pipe.ends[1] -> write(rBuf.begin(), rBuf.size());
				p2.wait(h2.waitScope());
			});
			
			auto p1 = pipe.ends[0] -> read(wBuf.begin(), wBuf.size(), wBuf.size());
			p1.wait(h1->waitScope());
		}
	}
	
	KJ_DBG("Check");
	REQUIRE(wBuf == rBuf);
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
