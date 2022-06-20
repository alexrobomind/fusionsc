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

class DefaultErrorHandler : public kj::TaskSet::ErrorHandler {
	void taskFailed(kj::Exception&& exception) override {
		KJ_LOG(WARNING, "Exception in connection", exception);
	}
};

struct MainCls {
	kj::ProcessContext& context;
	Maybe<int64_t> port = nullptr;
	kj::String address = kj::heapString("0.0.0.0");
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
	
	bool setAddress(kj::StringPtr val) {
		address = kj::str(val);
		return true;
	}
	
	bool setPort(kj::StringPtr val) {
		port = val.parseAs<int>();
		return true;
	}
		
	bool run() {
		auto l = newLibrary();
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
		
		Temporary<RootConfig> rootConfig;
		auto rootInterface = createRoot(lt, rootConfig);
		
		int port = 0;
		KJ_IF_MAYBE(pPort, this -> port)
			port = *pPort;
		
		// Get root network
		auto& network = lt->network();
		
		auto address = network.parseAddress(this->address, port).wait(ws);
		auto receiver = address->listen();
		
		std::cout << receiver->getPort() << std::endl;
		std::cout << std::endl;
		std::cout << "Listening on port " << receiver->getPort() << std::endl;
		
		DefaultErrorHandler errorHandler;
		kj::TaskSet tasks(errorHandler);
		
		// The accept handler runs in a recusrive loop and accepts new connections
		
		kj::Function<Promise<void>(Own<kj::AsyncIoStream>)> acceptHandler = [&](Own<kj::AsyncIoStream> connection) mutable -> Promise<void> {
			// Create RPC network
			auto vatNetwork = heapHeld<capnp::TwoPartyVatNetwork>(*connection, capnp::rpc::twoparty::Side::SERVER);
			
			// Initialize RPC system on top of network
			auto rpcSystem = heapHeld<capnp::RpcSystem<capnp::rpc::twoparty::VatId>>(capnp::makeRpcServer(*vatNetwork, rootInterface));
			
			// Run until the underlying connection disconnects
			auto task = vatNetwork->onDisconnect().attach(vatNetwork.x(), rpcSystem.x());
			tasks.add(mv(task));
			
			return receiver->accept().then(acceptHandler);
		};
		
		auto acceptLoop = receiver->accept().then(acceptHandler);
		
		auto shutdownPaf = kj::newPromiseAndCrossThreadFulfiller<void>();
		auto readThenFulfill = [&]() {
			std::cout << "Press any key to drain" << std::endl;
			std::string bla;
			std::getline(std::cin, bla);
			
			shutdownPaf.fulfiller -> fulfill();
		};
		
		kj::Thread cinReader(readThenFulfill);
		
		//TODO: On unix, listen for shutdown signals
		acceptLoop = acceptLoop.exclusiveJoin(mv(shutdownPaf.promise));
		
		acceptLoop.wait(ws);
		
		std::cout << "Waiting for clients to disconnect. Press Ctrl+C again to force shutdown." << std::endl;
		tasks.onEmpty().wait(ws);
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
