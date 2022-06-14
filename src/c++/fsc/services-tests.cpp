#include <catch2/catch_test_macros.hpp>

#include <kj/array.h>

#include "services.h"
#include "local.h"
#include "magnetics.h"
#include "geometry.h"

using namespace fsc;

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

TEST_CASE("resolverchain") {
	auto lib = newLibrary();
	auto t = lib->newThread();
	
	auto& ws = t->waitScope();
	
	// Construct dummy
	auto dummy = FieldResolver::Client(kj::heap<DummyResolver>());
	
	// Construct resolver chain
	auto chain = newResolverChain();
	
	for(size_t i = 0; i < 10; ++i ) {
		auto req = chain.registerRequest();
		req.setResolver(dummy);
		auto registration = req.send().getRegistration();
		
		fieldDummyCalled = false;
		auto field = chain.resolveFieldRequest().send().wait(ws);
		REQUIRE(fieldDummyCalled == true);
		REQUIRE(field.isInvert());
	}
	
	fieldDummyCalled = false;
	auto field = chain.resolveFieldRequest().send().wait(ws);
	REQUIRE(field.isSum());
	REQUIRE(fieldDummyCalled == false);
}