#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>

#include <capfuzz.h>

// Fallback that provides AFL macros if we are not compiling with AFL compiler
#include <capfuzz-miniafl.h>

#ifdef __AFL_COMPILER
#include <unistd.h>
#endif

using namespace fsc;

struct DataPublisher : public capfuzz::InputBuilder {
	constexpr static uint64_t DR_ID = capnp::typeId<fsc::DataRef<>>();
	
	static Maybe<capnp::StructSchema> payloadType(capnp::Type t) {
		if(!t.isInterface())
			return nullptr;
		
		auto schema = t.asInterface();
		if(schema.getProto().getId() != DR_ID)
			return nullptr;
		
		capnp::Type payload = schema.getBrandArgumentsAtScope(DR_ID)[0];
		if(!payload.isStruct())
			return nullptr;
		
		return payload.asStruct();
	}
	
	size_t getWeight(capnp::Type t) override {
		return payloadType(mv(t)) != nullptr ? 1 : 0;
	}
	
	inline capnp::Capability::Client getCapability(capnp::InterfaceSchema schema, Context& ctx) override {
		KJ_IF_MAYBE(pPayloadSchema, payloadType(schema)) {
			capnp::MallocMessageBuilder msg;
			auto root = msg.initRoot<capnp::DynamicStruct>(*pPayloadSchema);
			
			ctx.fillStruct(root);
			return getActiveThread().dataService().publish(root.asReader());
		}
		
		KJ_UNIMPLEMENTED();
	};
};

__AFL_FUZZ_INIT();

struct MainCls {
	kj::ProcessContext& context;
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
		
	bool run() {
		#ifdef __AFL_HAVE_MANUAL_CONTROL
			__AFL_INIT();
		#endif
		
		const unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;
		
		while (__AFL_LOOP(10000)) {
			int len = __AFL_FUZZ_TESTCASE_LEN;
			ArrayPtr<const byte> fuzzerData(buf, len);
			
			// Start fusionsc
			auto l = newLibrary();
			auto lt = l -> newThread();
			auto& ws = lt->waitScope();
			
			// Create backend with 1 thread and somewhat safe configuration
			Temporary<LocalConfig> config;
			config.getCpuBackend().getNumThreads().setFixed(1);
			config.getWorkerLauncher().setOff();
			config.setPreferredDeviceType(ComputationDeviceType::LOOP);
			
			RootService::Client rootService = createRoot(config.asReader());
			
			// Customize protocol
			DataPublisher dp;
			auto builders = kj::heapArray<capfuzz::InputBuilder*>({&dp});
			
			capfuzz::ProtocolConfig protoConfig;
			protoConfig.builders = builders.asPtr();
			
			// Set up targets
			auto targets = kj::heapArrayBuilder<capnp::DynamicCapability::Client>(1);
			targets.add(rootService);
			
			// Run fuzzer
			auto fuzzJob = capfuzz::runFuzzer(fuzzerData, targets.finish(), protoConfig).fork();
			
			// Wait until fuzzer job or timeout completes
			//fuzzJob.addBranch()
			//	.catch_([](kj::Exception&& e){})
			//	.exclusiveJoin(getActiveThread().timer().afterDelay(200 * kj::MILLISECONDS))
			//	.wait(ws);
			
			// Check whether fuzzer job completed
			if(fuzzJob.addBranch().poll(ws)) {
				// Extract exceptions
				fuzzJob.addBranch().wait(ws);
			}
		}

		return true;
	}
	
	auto getMain() {
		auto infoString = kj::str(
			"FusionSC fuzz test\n"
		);
		
		return kj::MainBuilder(context, infoString, "Runs the fuzz test. Accepts network input on stdin.")
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};


KJ_MAIN(MainCls)