#include "services.h"
#include "magnetics.h"
#include "kernels.h"
#include "flt.h"

#include <kj/list.h>

#include <capnp/rpc-twoparty.h>

using namespace fsc;

namespace {
	
template<typename T>
auto selectDevice(T t, WorkerType preferredType) {
	#ifdef FSC_WITH_CUDA
	
	try {
		if(preferredType == WorkerType::GPU) {
			return tuple(t(newGpuDevice()), WorkerType::GPU);
		}
	} catch(kj::Exception& e) {
	}
	
	#endif
	
	return tuple(t(newThreadPoolDevice()), WorkerType::CPU);
}

struct RootServer : public RootService::Server {
	RootServer(RootConfig::Reader config) {}
	
	Promise<void> newFieldCalculator(NewFieldCalculatorContext context) {
		auto factory = [this, context](auto device) mutable {		
		KJ_DBG(context.getParams());
			return ::fsc::newFieldCalculator(context.getParams().getGrid(), mv(device));
		};
		
		auto selectResult = selectDevice(factory, context.getParams().getPreferredDeviceType());
		
		auto results = context.getResults();
		results.setCalculator(kj::get<0>(selectResult));
		results.setDeviceType(kj::get<1>(selectResult));
		
		return READY_NOW;
	}
	
	Promise<void> newTracer(NewTracerContext context) {
		context.initResults().setService(newFLT(newThreadPoolDevice()));
		return READY_NOW;
	}
};

struct ResolverChainImpl : public virtual capnp::Capability::Server, public virtual ResolverChain::Server {
	using capnp::Capability::Server::DispatchCallResult;
	using ResolverChain::Server::RegisterContext;
	
	struct Registration {
		ResolverChainImpl& parent;
		kj::ListLink<Registration> link;
		
		capnp::Capability::Client entry;

		Registration(ResolverChainImpl& parent, capnp::Capability::Client entry);
		~Registration();
	};
	
	kj::List<Registration, &Registration::link> registrations;
	
	Promise<void> register_(RegisterContext ctx) override {		
		auto result = attach(
			ResolverChain::Client(capnp::newBrokenCap(KJ_EXCEPTION(UNIMPLEMENTED, "Unimplemented"))),
			
			thisCap(),
			kj::heap<Registration>(*this, ctx.getParams().getResolver())
		);
		ctx.initResults().setRegistration(mv(result));
		
		return READY_NOW;
	};
	
	DispatchCallResult dispatchCall(uint64_t interfaceId, uint16_t methodId, capnp::CallContext<capnp::AnyPointer, capnp::AnyPointer> ctx) override {
		Promise<void> result = ResolverChain::Server::dispatchCall(interfaceId, methodId, ctx).promise
		.catch_([=](kj::Exception&& exc) mutable -> Promise<void> {
			using capnp::AnyStruct;
			using capnp::AnyPointer;
			
			// We only handle generic methods if parent didnt have it implemented
			if(exc.getType() != kj::Exception::Type::UNIMPLEMENTED) {
				kj::throwRecoverableException(mv(exc));
				return READY_NOW;
			}
			
			// Updatable container on heap
			auto paramMsg = heapHeld<Own<capnp::MallocMessageBuilder>>();
			
			// Fill initial value from context
			*paramMsg = kj::heap<capnp::MallocMessageBuilder>();
			(**paramMsg).setRoot(ctx.getParams());
			ctx.releaseParams();
			
			// Check that first field is set
			{
				auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
				auto pSec = paramsStruct.getPointerSection();
				KJ_REQUIRE(pSec.size() > 0);
			}
			
			Promise<void> result = READY_NOW;
			for(auto& reg : registrations) {
				result = result.then([=, e = cp(reg.entry)]() mutable {
					auto params = (**paramMsg).getRoot<AnyPointer>();
					
					auto request = e.typelessRequest(
						interfaceId, methodId, params.targetSize()
					);
					request.set(params);
					
					return request.send();
				}).then([=](auto result) mutable {
					// Copy old extra parameters into new message, but drop old result
					{
						auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
						auto params       = (**paramMsg).getRoot<AnyPointer>();
						
						auto pSec = paramsStruct.getPointerSection();		
						pSec[0].clear();
						
						auto newParamMsg = kj::heap<capnp::MallocMessageBuilder>();
						newParamMsg->setRoot(params.asReader());
						*paramMsg = mv(newParamMsg);
					}
					
					// Copy result into params
					auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
					auto pSec = paramsStruct.getPointerSection();		
					pSec[0].set(result);
				}).catch_([](kj::Exception&& e) mutable {
					KJ_LOG(WARNING, "Exception in resolver chain", mv(e));
				});
			}
			
			result = result.then([=]() mutable {
				auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
				auto pSec = paramsStruct.getPointerSection();
				
				ctx.getResults().set(pSec[0]);
			});
			
			return result.attach(paramMsg.x(), thisCap());
		});
		
		return { mv(result), false };
	}
};

ResolverChainImpl::Registration::Registration(ResolverChainImpl& parent, capnp::Capability::Client entry) :
	parent(parent), entry(entry)
{
	parent.registrations.add(*this);
}

ResolverChainImpl::Registration::~Registration() {
	parent.registrations.remove(*this);
}

class DefaultErrorHandler : public kj::TaskSet::ErrorHandler {
	void taskFailed(kj::Exception&& exception) override {
		KJ_LOG(WARNING, "Exception in connection", exception);
	}
};

struct InProcessServerImpl {
	using Service = capnp::Capability;
	using Factory = kj::Function<Service::Client()>;
	
	Library library;
	Factory factory;
	kj::Thread thread;
	
	Own<const kj::Executor> executor;
	MutexGuarded<bool> ready = false;
	
	Own<CrossThreadPromiseFulfiller<void>> doneFulfiller;
		
	InProcessServer(const Library& library, kj::Function<capnp::Capability::Client()> factory) :
		library(library->addRef()),
		factory(factory),
		thread(KJ_BIND_METHOD(*this, run))
	{
		auto locked = ready.lock();
		locked.wait([](bool ready) { return ready; });
	}
	
	void run() {
		// Initialize event loop
		Library library = mv(this->library);
		auto lt = library -> newThread();
		auto& ws = lt -> waitScope();
		
		Promise<void> donePromise = READY_NOW;
		
		{
			auto locked = ready.lock();
			executor = kj::getCurrentThreadExecutor().addRef();
			*locked = true;
			
			auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
			doneFulfiller = mv(paf.fulfiller);
			donePromise = mv(paf.promise);
		}
		
		donePromise.wait(ws);
	}
	
	Service::Client accept() {
		auto pipe = getActiveThread().ioContext().provider->newTwoWayPipe();
		
		// Create server
		auto serverRunnable = [stream = mv(pipe.ends[1]), this]() {
			using capnp::TwoPartyVatNetwork;
			using capnp::rpc::twoparty::VatId;
			// Create RPC server on stream
			auto vatNetwork = heapHeld<capnp
		};
};

struct ServerImpl : public fsc::Server {
	Own<kj::ConnectionReceiver> receiver;
	DefaultErrorHandler errorHandler;
	
	kj::TaskSet tasks;
	
	Maybe<ForkedPromise<void>> drainPromise;
	
	kj::Canceler acceptLoopCanceler;
		
	RootService::Client rootInterface;
	
	ServerImpl(kj::NetworkAddress& address, RootService::Client rootInterface) :
		receiver(address.listen()),
		tasks(errorHandler),
		rootInterface(rootInterface)
	{}
	
	~ServerImpl() noexcept {}
	
	Promise<void> accept(Own<kj::AsyncIoStream> connection) {
		auto task = connection->write(MAGIC_TOKEN.begin(), MAGIC_TOKEN.size() - 1);
		
		task = task.then([&, connection=mv(connection)]() mutable {
			// Create RPC network
			auto vatNetwork = heapHeld<capnp::TwoPartyVatNetwork>(*connection, capnp::rpc::twoparty::Side::SERVER);
			
			// Initialize RPC system on top of network
			auto rpcSystem = heapHeld<capnp::RpcSystem<capnp::rpc::twoparty::VatId>>(capnp::makeRpcServer(*vatNetwork, rootInterface));
			
			// Run until the underlying connection disconnects
			return vatNetwork->onDisconnect().attach(vatNetwork.x(), rpcSystem.x(), mv(connection));
		});
		
		tasks.add(mv(task));
		
		auto result = receiver->accept().then([this](Own<kj::AsyncIoStream> connection) { return accept(mv(connection)); });
		result = acceptLoopCanceler.wrap(mv(result));
		return result;
	}
	
	unsigned int getPort() override { return receiver->getPort(); }
	
	Promise<void> run() override {
		return receiver->accept().then([this](Own<kj::AsyncIoStream> connection) { return accept(mv(connection)); });
	}
	
	Promise<void> drain() override {
		acceptLoopCanceler.cancel("Server draining");
		
		KJ_IF_MAYBE(pP, drainPromise) {
		} else {
			drainPromise = tasks.onEmpty().fork();
		}
		
		KJ_IF_MAYBE(pP, drainPromise)	{
			return pP->addBranch();
		}
		
		KJ_FAIL_REQUIRE("Internal error");
	}
};
	

}

RootService::Client fsc::createRoot(RootConfig::Reader config) {
	return kj::heap<RootServer>(config);
}

ResolverChain::Client fsc::newResolverChain() {
	return kj::heap<ResolverChainImpl>();
}

RootService::Client fsc::connectRemote(kj::StringPtr address, unsigned int portHint) {
	return getActiveThread().network().parseAddress(address, portHint)
	.then([](Own<kj::NetworkAddress> addr) {
		return addr->connect();
	})
	.then([](Own<kj::AsyncIoStream> connection) {
		auto tokenBuffer = kj::heapArray<byte>(MAGIC_TOKEN.size() - 1);
		auto task = connection->read(tokenBuffer.begin(), MAGIC_TOKEN.size() - 1);
		
		auto clientPromise = task.then([connection = mv(connection), tokenBuffer = mv(tokenBuffer)]() mutable {
			KJ_REQUIRE(tokenBuffer == MAGIC_TOKEN.asArray(), "Server returned invalid token");
			
			// Create RPC network
			auto vatNetwork = heapHeld<capnp::TwoPartyVatNetwork>(*connection, capnp::rpc::twoparty::Side::CLIENT);
			
			// Initialize RPC system on top of network
			auto rpcSystem = heapHeld<capnp::RpcSystem<capnp::rpc::twoparty::VatId>>(capnp::makeRpcClient(*vatNetwork));
			
			// Retrieve server's bootstrap interface
			Temporary<capnp::rpc::twoparty::VatId> serverID;
			serverID.setSide(capnp::rpc::twoparty::Side::SERVER);
			auto server = rpcSystem->bootstrap(serverID);
			
			return attach(server, rpcSystem.x(), vatNetwork.x(), mv(connection));
		});
		
		return capnp::Capability::Client(mv(clientPromise)).castAs<RootService>();
	});
}

Promise<Own<fsc::Server>> fsc::startServer(unsigned int portHint, kj::StringPtr address) {	
	Temporary<RootConfig> rootConfig;
	auto rootInterface = createRoot(rootConfig);
	
	// Get root network
	auto& network = getActiveThread().network();
	
	return network.parseAddress(address, portHint)
	.then([rootInterface](auto address) mutable -> Own<fsc::Server> {
		return kj::heap<ServerImpl>(*address, rootInterface);
	});
}