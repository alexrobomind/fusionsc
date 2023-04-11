#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>
#include <fsc/networking.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

using namespace fsc;

struct RootServiceProvider : public NetworkInterface::Listener::Server {
	Temporary<LocalConfig> config;
	
	RootServiceProvider(LocalConfig::Reader configIn) :
		config(configIn)
	{}
	
	Promise<void> accept(AcceptContext ctx) override {
		ctx.initResults().setClient(createRoot(config.asReader()));
		return READY_NOW;
	}
};

struct MainCls {
	kj::ProcessContext& context;
	Maybe<uint64_t> port = nullptr;
	kj::String address = kj::heapString("0.0.0.0");
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
	
	bool setAddress(kj::StringPtr val) {
		address = kj::str(val);
		return true;
	}
	
	bool setPort(kj::StringPtr val) {
		port = val.parseAs<uint64_t>();
		return true;
	}
		
	bool run() {
		auto l = newLibrary();
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
		
		NetworkInterface::Client networkInterface = kj::heap<LocalNetworkInterface>();
		
		Temporary<LocalConfig> config;
		
		auto listenRequest = networkInterface.listenRequest();
		KJ_IF_MAYBE(pPort, port) {
			listenRequest.setPortHint(*pPort);
		}
		listenRequest.setHost(address);
		listenRequest.setListener(kj::heap<RootServiceProvider>(config));
		
		auto openPort = listenRequest.sendForPipeline().getOpenPort();
		auto info = openPort.getInfoRequest().send().wait(ws);
		
		std::cout << "Listening on port " << info.getPort() << std::endl;
		
		while(true) {
			try {
				lt -> timer().afterDelay(1 * kj::SECONDS).wait(ws);
			} catch(kj::Exception e) {
				KJ_DBG("Received exception", e);
				break;
			}
		}
		
		std::cout << "Draining ..." << std::endl;
		openPort.drainRequest().send().wait(ws);
		
		std::cout << "Shutting down" << std::endl;		
		return true;
	}
	
	auto getMain() {
		return kj::MainBuilder(context, "FSC server", "Creates an FSC server")
			.addOptionWithArg({"-a", "address"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address to listen on, defaults to 0.0.0.0")
			.addOptionWithArg({"-p", "port"}, KJ_BIND_METHOD(*this, setPort), "<port>", "Port to listen on, defaults to system-assigned")
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};

KJ_MAIN(MainCls)
