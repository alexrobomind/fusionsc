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
#include "load-limiter.h"

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
			return mv(result);
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
		Warehouse::Folder::Client root = response.getRemote().template castAs<Warehouse::Folder>();
		
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
			return mv(result);
		}
	});
}

struct RootServer : public RootService::Server {
	RootServer(LocalConfig::Reader config) :
		config(config),
		device(selectDevice(config)),
		limiter(config.getLoadLimit())
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
	LoadLimiter limiter;
	
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
		
		return limiter.getToken()
		.then([this, context](auto token) mutable {
			context.initResults().setService(::fsc::newFieldCalculator(device -> addRef()).attach(mv(token)));
		});
	}
	
	Promise<void> newTracer(NewTracerContext context) override {		
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		return limiter.getToken()
		.then([this, context](auto token) mutable {
			context.initResults().setService(newFLT(device -> addRef(), config.getFlt()).attach(mv(token)));
		});
	}
	
	Promise<void> newGeometryLib(NewGeometryLibContext context) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		return limiter.getToken()
		.then([this, context](auto token) mutable {
			context.initResults().setService(fsc::newGeometryLib(device -> addRef()).attach(mv(token)));
		});
	}
	
	Promise<void> newHFCamProvider(NewHFCamProviderContext context) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		return limiter.getToken()
		.then([this, context](auto token) mutable {
			context.initResults().setService(fsc::newHFCamProvider(device -> addRef()).attach(mv(token)));
		});
	}
	
	Promise<void> newKDTreeService(NewKDTreeServiceContext context) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		return limiter.getToken()
		.then([this, context](auto token) mutable {
			context.initResults().setService(fsc::newKDTreeService().attach(mv(token)));
		});
	}
	
	Promise<void> newMapper(NewMapperContext ctx) override {
		KJ_REQUIRE(config.getEnableCompute(), "Computation is disabled on this node");
		
		auto flt = thisCap().newTracerRequest().sendForPipeline().getService();
		
		// Mapper holds token through tracer
		ctx.initResults().setService(fsc::newMapper(mv(flt), fsc::newKDTreeService(), fsc::newGeometryLib(device -> addRef()), *device));
		return READY_NOW;
	}
	
	Promise<void> newVmecDriver(NewVmecDriverContext ctx) override {
		return limiter.getToken()
		.then([this, ctx](auto token) mutable {
			ctx.initResults().setService(::fsc::createVmecDriver(device -> addRef(), selectScheduler()).attach(mv(token)));
		});
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
		
		results.setActiveCalls(limiter.getActive());
		results.setQueuedCalls(limiter.getQueued());
		results.setCapacity(limiter.getCapacity());
			
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
	
	kj::HashMap<uint64_t, DataRef<capnp::AnyPointer>::Client> store;
	uint64_t storeCounter = 0;
	
	LocalResourcesImpl(LocalConfig::Reader config) :
		rootService(createRoot(config))
	{}
	
	// Store implementation
	
	Promise<void> put(PutContext ctx) {
		auto params = ctx.getParams();
		
		const uint64_t id = storeCounter++;
		
		ctx.initResults().setId(id);
		
		if(!params.getDownload()) {
			store.insert(id, params.getRef());
			return READY_NOW; 
		}
		
		return getActiveThread().dataService().download(params.getRef(), /* recursive = */ true)
		.then([id, this](LocalDataRef<capnp::AnyPointer> ref) {
			store.insert(id, ref);
		});
	}
	
	Promise<void> get(GetContext ctx) {
		const uint64_t id = ctx.getParams().getId();
		KJ_IF_MAYBE(pValue, store.find(id)) {
			ctx.initResults().setRef(*pValue);
			return READY_NOW;
		}
		
		KJ_FAIL_REQUIRE("No entry found", id);
	}
	
	Promise<void> erase(EraseContext ctx) {
		store.erase(ctx.getParams().getId());
		return READY_NOW;
	}
	
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

}

kj::ArrayPtr<uint64_t> fsc::protectedInterfaces() {
	static kj::Array<uint64_t> result = kj::heapArray<uint64_t>({
		capnp::typeId<LocalResources>(),
		capnp::typeId<NetworkInterface>(),
		capnp::typeId<SSHConnection>(),
		capnp::typeId<LocalStore>()
	});
	
	return result.asPtr();
}

Own<LocalResources::Server> fsc::createLocalResources(LocalConfig::Reader config) {
	return kj::heap<LocalResourcesImpl>(config);
}

Own<RootService::Server> fsc::createRoot(LocalConfig::Reader config) {
	return kj::heap<RootServer>(config);
}
