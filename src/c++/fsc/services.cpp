#include "services.h"
#include "magnetics.h"

#include "geometry.h"
#include "flt.h"
#include "hfcam.h"
#include "fieldline-mapping.h"
#include "local-vat-network.h"
#include "ssh.h"
#include "networking.h"
#include "jobs.h"
#include "matcher.h"
#include "vmec.h"

#include "odb.h"
#include "sqlite.h"

#include "devices/w7x.h"

#include "kernels/device.h"

#include "commit-hash.h"

#include <capnp/rpc-twoparty.h>
#include <capnp/membrane.h>

#include <kj/list.h>
#include <kj/compat/url.h>
#include <kj/string-tree.h>
#include <kj/map.h>


#include <fsc/jobs.capnp.h>

#include <map>

using namespace fsc;

namespace {
	
Own<DeviceBase> selectDevice(LocalConfig::Reader config) {
	#ifdef FSC_WITH_CUDA
	
	try {
		if(config.getPreferredDeviceType() == ComputationDeviceType::GPU) {
			return kj::refcounted<GPUDevice>();
		}
	} catch(kj::Exception& e) {
	}
	
	#endif
	
	if(config.getPreferredDeviceType() == ComputationDeviceType::LOOP) {
		return LoopDevice::create();
	}
	
	auto numThreadsRequested = config.getCpuBackend().getNumThreads();
	
	uint32_t numThreads = (uint32_t) CPUDevice::estimateNumThreads();
	if(numThreadsRequested.isFixed()) {
		numThreads = numThreadsRequested.getFixed();
	}
	
	KJ_LOG(INFO, "Creating CPU backend", numThreads);
	return CPUDevice::create(numThreads);
}

Promise<Temporary<Warehouse::StoredObject>> connectWarehouse(kj::StringPtr urlString, NetworkInterface::Client networkInterface) {
	kj::UrlOptions opts;
	opts.allowEmpty = false;
	
	auto baseUrl = kj::Url::parse("sqlite://.");
	auto url = baseUrl.parseRelative(urlString);
	
	KJ_REQUIRE(
		url.scheme == "sqlite" || url.scheme == "http" || url.scheme == "ws",
		
		"Only the schemes 'sqlite' (local sqlite database file), 'ws' (http/websocket"
		" connection to remote DB server) or 'http' (alias for 'ws') are supported",
		url.scheme
	);
		
	if(url.scheme == "sqlite") {			
		bool readOnly = false;
		kj::StringPtr tablePrefix = "warehouse";
		kj::StringPtr rootName = "root";
		
		for(auto& param : url.query) {
			if(param.name == "readOnly")
				readOnly = true;
			
			if(param.name == "tablePrefix")
				tablePrefix = param.value;
		}
		
		kj::Path path(url.path.releaseAsArray());
		bool isAbsolute = url.host != ".";
				
		auto conn = connectSqlite(path.toNativeString(isAbsolute), readOnly);
		auto db = ::fsc::openWarehouse(*conn, readOnly, tablePrefix);
		
		auto req = db.getRootRequest();
		req.setName(rootName); // Not neccessary for main root
		
		auto root = req.sendForPipeline().getRoot();
		
		KJ_IF_MAYBE(pFragment, url.fragment) {
			auto getRequest = root.getRequest();
			getRequest.setPath(*pFragment);
			
			return getRequest.send().then([](auto response) {
				return Temporary<Warehouse::StoredObject>((Warehouse::StoredObject::Reader) response);
			});
		} else {
			Temporary<Warehouse::StoredObject> result;
			result.setFolder(root);
			result.setAsGeneric(root.castAs<Warehouse::GenericObject>());
			return result;
		}
	}
	
	KJ_REQUIRE(url.host != ".", "Must specify authority (hostname) in remote URLs");
	
	// Treat URL as server connection
	
	// Pass connect request to interface
	auto connectRequest = networkInterface.connectRequest();
	connectRequest.setUrl(urlString);
	
	// Open connection and get remote object
	return connectRequest.send()
	.then([](auto response) {
		return response.getConnection().getRemoteRequest().send();
	})
	.then([maybeFragment = mv(url.fragment)](auto response) -> Promise<Temporary<Warehouse::StoredObject>> {		
		auto root = response.getRemote().template castAs<Warehouse::Folder>();
		
		KJ_IF_MAYBE(pFragment, maybeFragment) {
			auto getRequest = root.getRequest();
			getRequest.setPath(*pFragment);
			
			return getRequest.send().then([](auto response) {
				return Temporary<Warehouse::StoredObject>((Warehouse::StoredObject::Reader) response);
			});
		} else {
			Temporary<Warehouse::StoredObject> result;
			result.setFolder(root);
			result.setAsGeneric(root.castAs<Warehouse::GenericObject>());
			return result;
		}
	});
}

struct RootServer : public RootService::Server {
	RootServer(LocalConfig::Reader config) :
		config(config),
		device(selectDevice(config))
	{
		NetworkInterface::Client nif = kj::heap<LocalNetworkInterface>();
		
		for(auto entry : config.getWarehouses()) {
			Warehouse::Folder::Client root = connectWarehouse(entry.getUrl(), nif)
			.then([name = entry.getName()](auto storedObject) {
				if(!storedObject.isFolder()) {
					KJ_LOG(ERROR, "Failed to open backend warehouse, connect succeeded but did not return a folder object", name, storedObject);
					KJ_FAIL_REQUIRE("Backend is not a folder");
				}
				
				KJ_REQUIRE(storedObject.isFolder(), "Invalid backend, not a folder");
				return storedObject.getFolder();
			});
			auto getReq = root.getRequest();
			getReq.setPath(entry.getPath());
			
			Warehouse::Folder::Client actualRoot = getReq.sendForPipeline().getAsGeneric();
			warehouses.insert(kj::heapString(entry.getName()), mv(actualRoot));
		}
	}
	
	Temporary<LocalConfig> config;
	Own<DeviceBase> device;
	
	Matcher::Client myMatcher = newMatcher();
	
	kj::TreeMap<kj::String, Warehouse::Folder::Client> warehouses;
	
	Own<JobLauncher> selectScheduler() {
		// Select correct scheduler
		switch(config.getJobScheduler().which()) {
			case LocalConfig::JobScheduler::SYSTEM:
				return newProcessScheduler(config.getJobDir());
			case LocalConfig::JobScheduler::SLURM:
				return newSlurmScheduler(newProcessScheduler(config.getJobDir()));
			case LocalConfig::JobScheduler::MPI:
				return newMpiScheduler(newProcessScheduler(config.getJobDir()));
			default:
				KJ_UNIMPLEMENTED("Unknown scheduler type requested.");
		}
	}
	
	Promise<void> newFieldCalculator(NewFieldCalculatorContext context) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		context.initResults().setService(::fsc::newFieldCalculator(device -> addRef()));		
		return READY_NOW;
	}
	
	Promise<void> newTracer(NewTracerContext context) override {		
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");	
		
		context.initResults().setService(newFLT(device -> addRef(), config.getFlt()));		
		return READY_NOW;
	}
	
	Promise<void> newGeometryLib(NewGeometryLibContext context) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		context.initResults().setService(fsc::newGeometryLib());
		return READY_NOW;
	}
	
	Promise<void> newHFCamProvider(NewHFCamProviderContext context) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		context.initResults().setService(fsc::newHFCamProvider());
		return READY_NOW;
	}
	
	Promise<void> newKDTreeService(NewKDTreeServiceContext context) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		context.initResults().setService(fsc::newKDTreeService());
		return READY_NOW;
	}
	
	Promise<void> newMapper(NewMapperContext ctx) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		auto flt = thisCap().newTracerRequest().sendForPipeline().getService();
		auto idx = thisCap().newKDTreeServiceRequest().sendForPipeline().getService();
		
		ctx.initResults().setService(fsc::newMapper(mv(flt), mv(idx), *device));
		return READY_NOW;
	}
	
	Promise<void> newVmecDriver(NewVmecDriverContext ctx) override {
		ctx.initResults().setService(::fsc::createVmecDriver(device -> addRef(), selectScheduler()));
		
		return READY_NOW;
	}
	
	Promise<void> dataService(DataServiceContext ctx) override {
		ctx.getResults().setService(getActiveThread().dataService());
		return READY_NOW;
	}
	
	Promise<void> getInfo(GetInfoContext ctx) override {
		auto results = ctx.getResults();
		if(device -> brand == &CPUDevice::BRAND) {
			results.setDeviceType(ComputationDeviceType::CPU);
		#ifdef FSC_WITH_CUDA
		} else if(device -> brand == &GPUDevice::BRAND) {
			results.setDeviceType(ComputationDeviceType::GPU);
		#endif
		} else {
			KJ_FAIL_REQUIRE("Unknown device type");
		}
		
		results.setComputeEnabled(config.getEnableCompute());
		
		{
			auto wh = results.initWarehouses(warehouses.size());
			size_t i = 0;
			for(auto& e : warehouses) {
				wh.set(i++, e.key);
			}
		}
		
		results.setName(config.getName());
		
		results.setCommitHash(::fsc::commitHash);
			
		return READY_NOW;
	}
	
	Promise<void> matcher(MatcherContext ctx) override {
		ctx.initResults().setService(myMatcher);
		return READY_NOW;
	}
	
	Promise<void> listWarehouses(ListWarehousesContext ctx) override {
		auto out = ctx.getResults().initNames(warehouses.size());
		
		size_t i = 0;
		for(auto& e : warehouses) {
			out.set(i++, e.key);
		}
		return READY_NOW;
	}
	
	Promise<void> getWarehouse(GetWarehouseContext ctx) override {
		auto name = kj::heapString(ctx.getParams().getName());
		
		KJ_IF_MAYBE(pWh, warehouses.find(name)) {
			ctx.getResults().setWarehouse(*pWh);
		} else {
			KJ_FAIL_REQUIRE("Warehouse not found", name);
		}
		
		return READY_NOW;
	}
};

// Networking implementation


struct LocalResourcesImpl : public LocalResources::Server, public LocalNetworkInterface {
	RootService::Client rootService;
	// Temporary<LocalConfig> config;
	
	LocalResourcesImpl(LocalConfig::Reader config) :
		rootService(createRoot(config))
	{}	
	
	// Root service
	
	Promise<void> root(RootContext ctx) override {
		ctx.getResults().setRoot(rootService);
		return READY_NOW;
	}
	
	Promise<void> configureRoot(ConfigureRootContext ctx) override {
		rootService = createRoot(ctx.getParams());
		return READY_NOW;
	}
	
	// File system access
	
	Promise<void> openArchive(OpenArchiveContext ctx) override {
		auto fs = kj::newDiskFilesystem();
		auto currentPath = fs -> getCurrentPath();
		auto realPath = currentPath.evalNative(ctx.getParams().getFilename());
		
		Own<const kj::ReadableFile> file = fs -> getRoot().openFile(realPath);
		auto result = getActiveThread().dataService().publishArchive<capnp::AnyPointer>(*file);
		
		ctx.getResults().setRef(mv(result));
		return READY_NOW;
	}
	
	Promise<void> writeArchive(WriteArchiveContext ctx) override {
		auto fs = kj::newDiskFilesystem();
		auto currentPath = fs -> getCurrentPath();
		auto realPath = currentPath.evalNative(ctx.getParams().getFilename());
		
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
	
	// Device providers
	
	Promise<void> w7xProvider(W7xProviderContext ctx) override {
		ctx.initResults().setService(devices::w7x::newProvider());
		return READY_NOW;
	}
	
	// Warehouse access
	
	Promise<void> openWarehouse(OpenWarehouseContext ctx) override {
		auto params = ctx.getParams();
		
		// Select connection interface to use (allows SSH tunnel)
		auto networkInterface =
			params.hasNetworkInterface() ?
			params.getNetworkInterface() :
			thisCap();
		
		return connectWarehouse(params.getUrl(), networkInterface)
		.then([ctx](Temporary<Warehouse::StoredObject> so) mutable {
			ctx.getResults().setObject(so.getAsGeneric());
			ctx.getResults().setStoredObject(so);
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
		
		vatHub(kj::heap<LocalVatHub>()),
		vatNetwork(vatHub -> join()),
		
		ready(false),
		thread(KJ_BIND_METHOD(*this, run))
	{
		auto locked = ready.lockExclusive();
		locked.wait([](bool ready) { return ready; });
		// thread.detach();
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
		ForkedPromise<void> lifetime;
		KeepaliveMembrane(Promise<void> lifetime) :
			lifetime(lifetime.fork())
		{}
		
		Own<MembranePolicy> addRef() override { return kj::addRef(*this); }
		
		kj::Maybe<capnp::Capability::Client> inboundCall(uint64_t interfaceId, uint16_t methodId, capnp::Capability::Client target) override {
			return nullptr;
		}
		
		kj::Maybe<capnp::Capability::Client> outboundCall(uint64_t interfaceId, uint16_t methodId, capnp::Capability::Client target) override {
			return nullptr;
		}
		
		kj::Maybe<Promise<void>> onRevoked() override {
			return lifetime.addBranch();
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
		
		auto vatNetwork = ownHeld(vatHub -> join());
		auto rpcClient  = heapHeld<capnp::RpcSystem<VatId>>(*vatNetwork, nullptr);
		auto client     = rpcClient -> bootstrap(LocalVatNetwork::INITIAL_VAT_ID);
		
		Own<void> attachments = kj::attachRef(client, vatNetwork.x(), rpcClient.x());
		Promise<void> lifetimeScope = getActiveThread().lifetimeScope().wrap(Promise<void>(NEVER_DONE)).attach(mv(attachments));
		return capnp::membrane(mv(client), kj::refcounted<KeepaliveMembrane>(mv(lifetimeScope)));
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

kj::ArrayPtr<uint64_t> fsc::protectedInterfaces() {
	static kj::Array<uint64_t> result = kj::heapArray<uint64_t>({
		capnp::typeId<LocalResources>(),
		capnp::typeId<NetworkInterface>(),
		capnp::typeId<SSHConnection>()
	});
	
	return result.asPtr();
}

kj::Function<capnp::Capability::Client()> fsc::newInProcessServer(kj::Function<capnp::Capability::Client()> serviceFactory) {
	auto server = kj::atomicRefcounted<InProcessServerImpl>(mv(serviceFactory));
	
	return [server = mv(server)]() mutable {
		return server->connect();
	};
}

Own<LocalResources::Server> fsc::createLocalResources(LocalConfig::Reader config) {
	return kj::heap<LocalResourcesImpl>(config);
}

Own<RootService::Server> fsc::createRoot(LocalConfig::Reader config) {
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
	Temporary<LocalConfig> rootConfig;
	RootService::Client rootInterface = createRoot(rootConfig);
	
	// Get root network
	auto& network = getActiveThread().network();
	
	return network.parseAddress(address, portHint)
	.then([rootInterface](auto address) mutable -> Own<fsc::Server> {
		return kj::heap<ServerImpl>(*address, rootInterface);
	});
}