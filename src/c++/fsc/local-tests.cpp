#include <catch2/catch_test_macros.hpp>

#include <kj/thread.h>
#include <kj/string.h>

#include "local.h"

using namespace fsc;

TEST_CASE("local-setup") {
	kj::Own<const Library> lib = Library::create();
	
	SECTION("noop") {
	}
	
	SECTION("addref", "Can add refs") {
		kj::Own<const Library> lib2 = lib -> addRef();
		SUCCEED();
	}
	
	SECTION("th", "Can have a thread handle") {
		ThreadHandle th(lib);
		SUCCEED();
		
		SECTION("randworks") {
			kj::FixedArray<byte, 128> a1;
			th.rng().randomize(a1);
			
			SUCCEED();
		}
	}
	
	SECTION("no2th", "Can not have more than one thread handle") {
		ThreadHandle th1(lib);
		
		REQUIRE_THROWS(ThreadHandle(lib));
	}
}

TEST_CASE("threading") {
	kj::Own<const Library> lib = Library::create();
	
	ThreadHandle h1(lib);
	
	SECTION("launchout", "Can open a second handle in another thread") {
		{
			kj::Thread t([&]() {
				ThreadHandle h2(lib);
			});
		}
		SUCCEED();
	}
	
	SECTION("connect", "Can connect two threads (repeats multiple times)") {
	for(size_t i = 0; i < 10; ++i) { DYNAMIC_SECTION("repeat" << i) {
		CrossThreadConnection conn;
		
		kj::FixedArray<byte, 4> recvBuf;
		kj::FixedArray<byte, 4> sendBuf;
		
		h1.rng().randomize(sendBuf);
		
		SECTION("mainaccepts") {
			kj::Thread t([&lib, &conn, &sendBuf]() {
				ThreadHandle h2(lib);
				auto streamPromise = conn.connect(h2);
				auto stream = streamPromise.wait(h2.waitScope());
				
				stream -> write(sendBuf.begin(), sendBuf.size()).wait(h2.waitScope());
			});
		
			auto streamPromise = conn.accept(h1);
			auto stream = streamPromise.wait(h1.waitScope());
			
			stream -> read(recvBuf.begin(), recvBuf.size()).wait(h1.waitScope());
	
			REQUIRE((ArrayPtr<byte>) sendBuf == (ArrayPtr<byte>) recvBuf);
		}
		
		SECTION("mainconnects") {
			kj::Thread t([&lib, &conn, &sendBuf]() {
				ThreadHandle h2(lib);
				auto streamPromise = conn.accept(h2);
				auto stream = streamPromise.wait(h2.waitScope());
				
				stream -> write(sendBuf.begin(), sendBuf.size()).wait(h2.waitScope());
			});
		
			auto streamPromise = conn.connect(h1);
			auto stream = streamPromise.wait(h1.waitScope());
			
			stream -> read(recvBuf.begin(), recvBuf.size()).wait(h1.waitScope());
	
			REQUIRE((ArrayPtr<byte>) sendBuf == (ArrayPtr<byte>) recvBuf);
		}
	}}
	}
	
	SECTION("selfconnect", "Can self connect") {
		CrossThreadConnection conn;
		
		SECTION("accept_then_connect") {
			auto streamAcc = conn.accept(h1);
			auto streamConn = conn.connect(h1);
			
			SECTION("wait_accept_first") {
				streamAcc.wait(h1.waitScope());
				streamConn.wait(h1.waitScope());
				SUCCEED();
			}
			
			SECTION("wait_connect_first") {
				streamConn.wait(h1.waitScope());
				streamAcc.wait(h1.waitScope());
				SUCCEED();
			}
		}
		
		SECTION("connect_then_accept") {
			auto streamConn = conn.connect(h1);
			auto streamAcc = conn.accept(h1);
			
			SECTION("wait_accept_first") {
				streamAcc.wait(h1.waitScope());
				streamConn.wait(h1.waitScope());
				SUCCEED();
			}
			
			SECTION("wait_connect_first") {
				streamConn.wait(h1.waitScope());
				streamAcc.wait(h1.waitScope());
				SUCCEED();
			}
		}
	}
}