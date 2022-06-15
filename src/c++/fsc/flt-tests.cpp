#include <catch2/catch_test_macros.hpp>

#include <kj/array.h>

#include "services.h"
#include "local.h"
#include "magnetics.h"
#include "geometry.h"

using namespace fsc;

TEST_CASE("flt") {
	auto l = newLibrary();
	auto lt = l->newThread();
	
	auto& ws = lt->waitScope();
	
	Temporary<RootConfig> config;
	
	auto req = createRoot(lt, config).newTracerRequest();
	auto flt = req.send().getService();
}