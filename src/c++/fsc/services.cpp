#include "services.h"
#include "magnetics.h"
#include "kernels.h"
#include "geometry.h"
#include "flt.h"
#include "hfcam.h"
#include "index.h"
#include "fieldline-mapping.h"
#include "local-vat-network.h"
#include "ssh.h"

#include <capnp/rpc-twoparty.h>
#include <capnp/membrane.h>

#include <kj/list.h>

using namespace fsc;

namespace {
	
kj::Tuple<Own<DeviceBase>, WorkerType> selectDevice(WorkerType preferredType) {
	#ifdef FSC_WITH_CUDA
	
	try {
		if(preferredType == WorkerType::GPU) {
			return kj:tuple(kj::refcounted<GPUDevice>(), WorkerType::GPU);
		}
	} catch(kj::Exception& e) {
	}
	
	#endif
	
	return kj::tuple(kj::refcounted<CPUDevice>(), WorkerType::CPU);
}

struct RootServer : public RootService::Server {
	RootServer(RootConfig::Reader config) {}
	
	Promise<void> newFieldCalculator(NewFieldCalculatorContext context) override {
		auto selectResult = selectDevice(context.getParams().getPreferredDeviceType());
		
		auto results = context.getResults();
		results.setService(::fsc::newFieldCalculator(mv(kj::get<0>(selectResult))));
		results.setDeviceType(kj::get<1>(selectResult));
		
		return READY_NOW;
	}
	
	Promise<void> newTracer(NewTracerContext context) override {		
		auto selectResult = selectDevice(context.getParams().getPreferredDeviceType());
		
		auto results = context.getResults();
		results.setService(::fsc::newFLT(mv(kj::get<0>(selectResult))));
		results.setDeviceType(kj::get<1>(selectResult));
		
		return READY_NOW;
	}
	
	Promise<void> newGeometryLib(NewGeometryLibContext context) override {
		context.initResults().setService(fsc::newGeometryLib());
		return READY_NOW;
	}
	
	Promise<void> newHFCamProvider(NewHFCamProviderContext context) override {
		context.initResults().setService(fsc::newHFCamProvider());
		return READY_NOW;
	}
	
	Promise<void> newKDTreeService(NewKDTreeServiceContext context) override {
		context.initResults().setService(fsc::newKDTreeService());
		return READY_NOW;
	}
	
	Promise<void> newMapper(NewMapperContext ctx) override {
		auto flt = thisCap().newTracerRequest().send().getService();
		auto idx = thisCap().newKDTreeServiceRequest().send().getService();
		ctx.initResults().setService(fsc::newMapper(mv(flt), mv(idx)));
		return READY_NOW;
	}
	
	Promise<void> dataService(DataServiceContext ctx) override {
		ctx.getResults().setService(getActiveThread().dataService());
		return READY_NOW;
	}
};

// Networking implementation

struct NetworkInterfaceBase : public virtual NetworkInterface::Server {
	virtual Promise<Own<kj::AsyncIoStream>> makeConnection(kj::StringPtr host, unsigned int port) = 0;
	
	Promise<void> sshConnect(SshConnectContext ctx);
};

struct SSHConnectionImpl : public SSHConnection::Server, public NetworkInterfaceBase {
	Own<SSHSession> session;
	
	SSHConnectionImpl(Own<SSHSession> session) :
		session(mv(session))
	{}
	
	Promise<Own<kj::AsyncIoStream>> makeConnection(kj::StringPtr host, unsigned int port) override {
		KJ_REQUIRE(session -> isAuthenticated(), "Not authenticated");
		return session -> connectRemote(host, port)
		.then([](Own<SSHChannel> channel) {
			return channel -> openStream(0);
		});
	}
	
	Promise<void> close(CloseContext ctx) {
		session -> close();
		return session -> drain();
	}
	
	Promise<void> authenticatePassword(AuthenticatePasswordContext ctx) {
		auto params = ctx.getParams();
		return session -> authenticatePassword(params.getUser(), params.getPassword())
		.then([](bool result) {
			KJ_REQUIRE(result, "Authentication failed");
		});
	}
};

Promise<void> NetworkInterfaceBase::sshConnect(SshConnectContext ctx) {
	auto params = ctx.getParams();
	return makeConnection(params.getHost(), params.getPort())
	.then([](Own<kj::AsyncIoStream> stream) {
		return createSSHSession(mv(stream));
	})
	.then([ctx](Own<SSHSession> sshSession) mutable {
		ctx.initResults().setConnection(kj::heap<SSHConnectionImpl>(mv(sshSession)));
	});
}

struct LocalResourcesImpl : public LocalResources::Server, public NetworkInterfaceBase {
	Temporary<RootConfig> config;
	LocalResourcesImpl(RootConfig::Reader config) :
		config(config)
	{}
	
	// Local network interface
	
	virtual Promise<Own<kj::AsyncIoStream>> makeConnection(kj::StringPtr host, unsigned int port) {
		return getActiveThread().network().parseAddress(host, port)
		.then([](Own<kj::NetworkAddress> addr) {
			return addr -> connect();
		});
	}
	
	// Root service
	
	Promise<void> root(RootContext ctx) override {
		ctx.getResults().setRoot(createRoot(config));
		return READY_NOW;
	}
	
	// File system access
	
	Promise<void> openArchive(OpenArchiveContext ctx) override {
		auto fs = kj::newDiskFilesystem();
		auto currentPath = fs -> getCurrentPath();
		auto realPath = currentPath.eval(ctx.getParams().getFilename());
		
		Own<const kj::ReadableFile> file = fs -> getRoot().openFile(realPath);
		auto result = getActiveThread().dataService().publishArchive<capnp::AnyPointer>(*file);
		
		ctx.getResults().setRef(mv(result));
		return READY_NOW;
	}
	
	Promise<void> writeArchive(WriteArchiveContext ctx) override {
		auto fs = kj::newDiskFilesystem();
		auto currentPath = fs -> getCurrentPath();
		auto realPath = currentPath.eval(ctx.getParams().getFilename());
		
		Own<const kj::File> file = fs -> getRoot().openFile(realPath, kj::WriteMode::CREATE | kj::WriteMode::MODIFY | kj::WriteMode::CREATE_PARENT);
		Promise<void> result = getActiveThread().dataService().writeArchive(ctx.getParams().getRef(), *file);
		
		return result.attach(mv(file));
	}
	
	// Local data store access
	
	Promise<void> download(DownloadContext ctx) override {		
		return getActiveThread().dataService().download(ctx.getParams().getRef())
		.then([ctx](LocalDataRef<capnp::AnyPointer> ref) mutable {
			ctx.getResults().setRef(mv(ref));
		});
	}
};

class DefaultErrorHandler : public kj::TaskSet::ErrorHandler {
	void taskFailed(kj::Exception&& exception) override {
		KJ_LOG(WARNING, "Exception in connection", exception);
	}
};

struct InProcessServerImpl : public kj::AtomicRefcounted, public capnp::BootstrapFactory<lvn::VatId> {
	using Service = capnp::Capability;
	using Factory = kj::Function<Service::Client()>;
	using VatId = fsc::lvn::VatId;
	
	Library library;
	mutable Factory factory;
	
	Own<LocalVatHub> vatHub;
	Own<LocalVatNetwork> vatNetwork;
	
	kj::MutexGuarded<bool> ready;
		
	// The desctructor of this joins the inner runnable. Everything above
	// can be safely used from the inside.
	kj::Thread thread;
	
	// Own<const kj::Executor> executor;
	Own<CrossThreadPromiseFulfiller<void>> doneFulfiller;
		
	InProcessServerImpl(kj::Function<capnp::Capability::Client()> factory) :
		library(getActiveThread().library()->addRef()),
		factory(mv(factory)),
		
		vatHub(newLocalVatHub()),
		vatNetwork(kj::heap<LocalVatNetwork>(*vatHub)),
		
		ready(false),
		thread(KJ_BIND_METHOD(*this, run))
	{
		auto locked = ready.lockExclusive();
		locked.wait([](bool ready) { return ready; });
		thread.detach();
	}
	
	~InProcessServerImpl() {
		doneFulfiller->fulfill();
	}
	
	Own<const InProcessServerImpl> addRef() const { return kj::atomicAddRef(*this); }
	
	capnp::Capability::Client createFor(VatId::Reader clientId) {
		return factory();
	}
	
	//! Keep-alive membrane that maintains the connection as long as at least one instance is there
	struct KeepaliveMembrane : public capnp::MembranePolicy, kj::Refcounted {
		Own<void> keepAlive;
		KeepaliveMembrane(Own<void> keepAlive) : keepAlive(mv(keepAlive)) {}
		
		Own<MembranePolicy> addRef() override { return kj::addRef(*this); }
		
		kj::Maybe<capnp::Capability::Client> inboundCall(uint64_t interfaceId, uint16_t methodId, capnp::Capability::Client target) override {
			return nullptr;
		}
		
		kj::Maybe<capnp::Capability::Client> outboundCall(uint64_t interfaceId, uint16_t methodId, capnp::Capability::Client target) override {
			return nullptr;
		}
	};
	
	void run() {
		// Initialize event loop
		Library library = this->library->addRef();
		auto lt = library -> newThread();
		auto& ws = lt -> waitScope();
		
		// Create server
		using capnp::RpcSystem;
		using fsc::lvn::VatId;
		
		// Move vat network into local scope and shadow it
		Own<LocalVatNetwork> vatNetwork = mv(this -> vatNetwork);
		capnp::RpcSystem<VatId> rpcSystem(*vatNetwork, *this);
		
		Promise<void> donePromise = READY_NOW;
		
		{
			auto locked = ready.lockExclusive();
			// executor = kj::getCurrentThreadExecutor().addRef();
			
			auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
			doneFulfiller = mv(paf.fulfiller);
			donePromise = mv(paf.promise);
			
			*locked = true;
		}
		
		donePromise.wait(ws);
	}
	
	Service::Client connect() const {
		using capnp::RpcSystem;
		
		auto vatNetwork = heapHeld<LocalVatNetwork>(*vatHub);
		auto rpcClient  = heapHeld<capnp::RpcSystem<VatId>>(*vatNetwork, nullptr);
		auto client     = rpcClient -> bootstrap(LocalVatHub::INITIAL_VAT_ID);
		
		Own<void> attachments = kj::attachRef(client, vatNetwork.x(), rpcClient.x());
		return capnp::membrane(mv(client), kj::refcounted<KeepaliveMembrane>(mv(attachments)));
	}
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
			return vatNetwork->onDisconnect().attach(mv(connection), vatNetwork.x(), rpcSystem.x());
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


kj::Function<capnp::Capability::Client()> fsc::newInProcessServer(kj::Function<capnp::Capability::Client()> serviceFactory) {
	auto server = kj::atomicRefcounted<InProcessServerImpl>(mv(serviceFactory));
	
	return [server = mv(server)]() mutable {
		return server->connect();
	};
}

LocalResources::Client fsc::createLocalResources(RootConfig::Reader config) {
	return kj::heap<LocalResourcesImpl>(config);
}

RootService::Client fsc::createRoot(RootConfig::Reader config) {
	return kj::heap<RootServer>(config);
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