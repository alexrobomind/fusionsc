#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

using namespace fsc;

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
		
		unsigned int port = 0;
		KJ_IF_MAYBE(pPort, this -> port)
			port = *pPort;
			
		auto server = fsc::startServer(lt, port, this->address).wait(ws);
		port = server->getPort();
		
		std::cout << port << std::endl;
		std::cout << std::endl;
		std::cout << "Listening on port " << port << std::endl;
		
		auto shutdownPaf = kj::newPromiseAndCrossThreadFulfiller<void>();
		auto readThenFulfill = [&]() {
			std::cout << "Press any key to drain" << std::endl;
			std::string bla;
			std::getline(std::cin, bla);
			
			shutdownPaf.fulfiller -> fulfill();
		};
		
		kj::Thread cinReader(readThenFulfill);
		
		//TODO: On unix, listen for shutdown signals
		Promise<void> promise = server->run();
		promise = promise.exclusiveJoin(mv(shutdownPaf.promise));
		
		promise.wait(ws);
		
		std::cout << "Waiting for clients to disconnect. Press Ctrl+C again to force shutdown." << std::endl;
		server->drain().wait(ws);
		std::cout << "All clients disconnected. Good bye :-)" << std::endl;
		
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
