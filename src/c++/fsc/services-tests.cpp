#include <catch2/catch_test_macros.hpp>

#include <kj/array.h>

#include "services.h"
#include "local.h"
#include "magnetics.h"
#include "geometry.h"

using namespace fsc;

TEST_CASE("in-process-server") {
	auto library = newLibrary();
	auto thread = library -> newThread();
	
	auto localFactory = []() {
		Temporary<RootConfig> conf;
		return createLocalResources(conf);
	};
	
	auto server = newInProcessServer<LocalResources>(mv(localFactory));
	KJ_DBG("Server created. Connecting ...");
	
	auto connected = server();
	KJ_DBG("Connection formed, waiting for resolution ...");
	connected.whenResolved().wait(thread -> waitScope());
	KJ_DBG("Resolved");
}

TEST_CASE("http-connect") {
	auto library = newLibrary();
	auto thread = library -> newThread();
	auto& ws = thread -> waitScope();
	auto& ds = thread -> dataService();
	
	// Create a DataRef as service we want to serve
	
	Temporary<RootConfig> conf;
	auto lr = createLocalResources(conf);
	
	auto data = kj::heapArray<capnp::byte>(20);
	thread -> rng().randomize(data);
	auto published = ds.publish((capnp::Data::Reader) data);
	
	auto serveRequest = lr.serveRequest();
	serveRequest.setServer(published);
	auto openPort = serveRequest.send().wait(ws).getOpenPort();
	auto port = openPort.getInfoRequest().send().wait(ws).getPort();
	KJ_DBG(port);
	
	auto connReq = lr.connectRequest();
	connReq.setUrl(kj::str("http://localhost:", port));
	auto connection = connReq.send().wait(ws).getConnection();
	
	auto remote = connection.getRemoteRequest().send().wait(ws).getRemote();
	auto copy = ds.download(remote.castAs<DataRef<capnp::Data>>()).wait(ws);
	
	KJ_REQUIRE(copy.get() == data);
	
	remote = nullptr;
	connection = nullptr;
	
	openPort.drainRequest().send().wait(ws);
	
	KJ_DBG("Finished test");
}

TEST_CASE("ssh-connect", "[.][ssh]") {
	auto library = newLibrary();
	auto thread = library -> newThread();
	
	Temporary<RootConfig> config;
	auto localResources = createLocalResources(config.asReader());
	
	auto req = localResources.sshConnectRequest();
	req.setHost("localhost");
	req.setPort(2222);
	
	auto conn = req.send().wait(thread->waitScope()).getConnection();
	
	SECTION("badauth") {
		auto req = conn.authenticatePasswordRequest();
		req.setUser("testuser");
		req.setPassword("wrongPassword");
		REQUIRE_THROWS(req.send().wait(thread -> waitScope()));
	}
	
	{
		auto req = conn.authenticatePasswordRequest();
		req.setUser("testuser");
		req.setPassword("testpass");
		req.send().wait(thread -> waitScope());
	}

}

namespace {
	bool fieldDummyCalled;
	
	class DummyResolver : public FieldResolver::Server {
		using FieldResolver::Server::ResolveFieldContext;
		
		Promise<void> resolveField(ResolveFieldContext ctx) override {
			fieldDummyCalled = true;
			// ctx.setResults(ctx.getParams().getField());
			ctx.getResults().initInvert();
			return READY_NOW;
		}
	};
}