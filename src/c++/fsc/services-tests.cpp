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

TEST_CASE("ssh-connect", "[.][ssh]") {
	auto library = newLibrary();
	auto thread = library -> newThread();
	
	Temporary<RootConfig> config;
	auto localResources = createLocalResources(config.asReader());
	
	auto req = localResources.sshConnectRequest();
	req.setHost("localhost");
	req.setPort(2222);
	
	auto conn = req.send().wait(thread->waitScope()).getConnection();
	
	{
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