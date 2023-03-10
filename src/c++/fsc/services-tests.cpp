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