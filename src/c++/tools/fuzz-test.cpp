#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>
#include <fsc/networking.h>
#include <fsc/yaml.h>
#include <fsc/load-balancer.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

using namespace fsc;

struct DummyOutputStream : public kj::AsyncOutputStream {
	Promise<void> write(const void* buffer, size_t size) override {
		return READY_NOW;
	}
	
	Promise<void> write(kj::ArrayPtr<const kj::ArrayPtr<const byte>> pieces) override {
		return READY_NOW;
	}
	
	Promise<void> whenWriteDisconnected() {
		return NEVER_DONE;
	}
};

struct MainCls {
	kj::ProcessContext& context;
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
		
	bool run() {
		// Start fusionsc
		auto l = newLibrary(true);
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
		
		// Create backend with 1 thread and somewhat safe configuration
		Temporary<LocalConfig> config;
		config.getCpuBackend().getNumThreads().setFixed(1);
		config.getWorkerLauncher().setOff();
		
		RootService::Client rootService = createRoot(config.asReader());
		
		// Create data pipe
		auto pipe = kj::newTwoWayPipe();
		
		// stdio -> end 1
		auto standardInput = getActiveThread().ioContext().lowLevelProvider -> wrapInputFd(0);
		auto pumpLoop1 = standardInput -> pumpTo(*pipe.ends[0]).eagerlyEvaluate(nullptr);
		
		// end 1 -> dummy
		DummyOutputStream dummy;
		auto pumpLoop2 = pipe.ends[0] -> pumpTo(dummy).eagerlyEvaluate(nullptr);
		
		// Create server on other end
		capnp::TwoPartyVatNetwork vatNetwork(*pipe.ends[1], capnp::rpc::twoparty::Side::SERVER);
		capnp::RpcSystem<capnp::rpc::twoparty::VatId> rpcSystem(vatNetwork, rootService);
		
		rpcSystem.run().wait(ws);
		return true;
	}
	
	auto getMain() {
		auto infoString = kj::str(
			"FusionSC fuzz test\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Runs the fuzz test. Accepts network input on stdin.")
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};


KJ_MAIN(MainCls)