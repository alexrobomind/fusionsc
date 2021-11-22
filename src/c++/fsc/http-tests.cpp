#include <catch2/catch_test_macros.hpp>
#include <fsc/http.capnp.h>

#include <kj/debug.h>

#include "http.h"

namespace fsc {

TEST_CASE("http") {
	kj::_::Debug::setLogLevel(kj::LogSeverity::INFO);
	//EnableFullStackTrace fullTrace;
	
	using kj::HttpMethod;
	
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	kj::WaitScope& ws = th -> ioContext().waitScope;
	
	auto& network = th -> ioContext().provider -> getNetwork();
	
	kj::HttpHeaderTable hTbl;
	
	kj::HttpHeaders headers(hTbl);
	headers.add("Host", "localhost");
	
	SimpleHttpServer srv(
		network.parseAddress("127.0.0.1"),
		th,
		HTTP_TEST_DATA
	);
	
	auto& active = srv.listen();
	
	if(active.poll(ws))
		// Uh oh, something failed during port binding. Print error.
		active.wait(ws);
	
	unsigned int port = srv.getPort().wait(ws);
	KJ_LOG(WARNING, port);
	
	auto remoteAddr = network.parseAddress("127.0.0.1", port).wait(ws);
	KJ_LOG(WARNING, *remoteAddr);
	
	auto client = kj::newHttpClient(
		th -> ioContext().provider -> getTimer(),
		kj::HttpHeaderTable(),
		*remoteAddr
	);
	//auto client = kj::newHttpClient(srv);
	
	auto response = client->request(HttpMethod::GET, "/", headers).response.wait(ws);
	KJ_LOG(WARNING, "Got response");
	auto body = response.body->readAllText().wait(ws);
	KJ_LOG(WARNING, "Got body");
	
	REQUIRE(body == "Hello world");
}

}