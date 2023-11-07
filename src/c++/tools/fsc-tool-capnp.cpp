#include <fsc/local.h>

#include "fsc-tool.h"

using namespace fsc;

namespace {

struct CapnpId {
	kj::ProcessContext& context;
	
	CapnpId(kj::ProcessContext& context):
		context(context)
	{}
	
	bool run() {	
		auto lib = newLibrary();
		auto lt = lib -> newThread();
		
		uint64_t id;
		lt -> rng().randomize(kj::ArrayPtr<kj::byte>((kj::byte*) &id, sizeof(uint64_t)));
		
		context.exitInfo(kj::str("@0x", kj::hex(id)));
		return true;
	}
	
	auto getMain() {
		return kj::MainBuilder(context, "Capnp ID tool", "Generates an ID to use for .capnp.files")
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build()
		;
	}
};
	
struct CapnpTool {
	kj::ProcessContext& context;
	
	CapnpTool(kj::ProcessContext& context):
		context(context)
	{}
	
	auto getMain() {
		auto infoString = kj::str(
			"FusionSC capnp tool\n"
		);
		
		return kj::MainBuilder(context, infoString, "Capnproto tool")
			.addSubCommand("id", KJ_BIND_METHOD(CapnpId(context), getMain), "Proxy for cap'n'proto's capnp tool")
			.build()
		;
	}
};

}

fsc_tool::MainGen fsc_tool::capnp(kj::ProcessContext& ctx) {
	return KJ_BIND_METHOD(CapnpTool(ctx), getMain);
}