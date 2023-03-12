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