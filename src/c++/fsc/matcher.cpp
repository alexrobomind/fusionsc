#include "matcher.h"
#include "local.h"

#include <kj/map.h>

namespace fsc {

namespace {

struct MatcherImpl : public Matcher::Server {
	using F = Own<kj::PromiseFulfiller<capnp::Capability::Client>>;
	
	kj::TreeMap<ID, F> fulfillers;
	
	Promise<void> get(GetContext ctx) override {
		ID id = getActiveThread().randomID().asPtr();
		
		// Ensure that ID is unique
		while(fulfillers.find(id) != nullptr) {
			id = ID(getActiveThread().randomID());
		}
		
		auto paf = kj::newPromiseAndFulfiller<capnp::Capability::Client>();
		fulfillers.insert(id, mv(paf.fulfiller));
		
		auto res = ctx.initResults();
		res.setToken(id.asPtr());
		res.setCap(mv(paf.promise));
		
		return READY_NOW;
	}
	
	Promise<void> put(PutContext ctx) override {
		auto params = ctx.getParams();
		
		auto maybeFulfiller = fulfillers.find(params.getToken());
		
		KJ_IF_MAYBE(pFulfiller, maybeFulfiller) {
			(**pFulfiller).fulfill(params.getCap());
		} else {
			KJ_FAIL_REQUIRE("Token not found");
		}
		
		return READY_NOW;
	}
};

}

Own<Matcher::Server> newMatcher() {
	return kj::heap<MatcherImpl>();
}

}