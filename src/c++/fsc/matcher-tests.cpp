#include <catch2/catch_test_macros.hpp>

#include "matcher.h"
#include "local.h"

namespace fsc {

TEST_CASE("matcher") {
	using capnp::Capability;
	using capnp::CapabilityServerSet;
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	CapabilityServerSet<Matcher> serverSet;
	
	auto addr = [&](Matcher::Client clt) -> Matcher::Server* {
		auto maybeRef = serverSet.getLocalServer(clt).wait(ws);
		KJ_IF_MAYBE(pRef, maybeRef) {
			return &(*pRef);
		}
		
		KJ_FAIL_REQUIRE("Could not unwrap server");
	};
	
	Matcher::Client clt = serverSet.add(newMatcher());
	
	Matcher::Client matcher = newMatcher();
	
	auto getResponse = matcher.getRequest().send().wait(ws);
	
	auto putRequest = matcher.putRequest();
	putRequest.setToken(getResponse.getToken());
	putRequest.setCap(clt);
	putRequest.send().wait(ws);
	
	KJ_REQUIRE(addr(getResponse.getCap().castAs<Matcher>()) == addr(clt), "Incorrect client returned");
}

}