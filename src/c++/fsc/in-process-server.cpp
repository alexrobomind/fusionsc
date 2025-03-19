#include "in-process-server.h"

#include "memory.h"
#include "data.h"

#include <capnp/membrane.h>

namespace fsc { namespace {

struct InProcessServerImpl : public kj::AtomicRefcounted, public capnp::BootstrapFactory<lvn::VatId>, public InProcessServer {
	using Service = capnp::Capability;
	using Factory = kj::Function<Service::Client()>;
	using VatId = fsc::lvn::VatId;
	
	Library library;
	mutable Factory factory;
	
	LocalVatHub vatHub;
	
	kj::MutexGuarded<bool> ready;
		
	// The desctructor of this joins the inner runnable. Everything above
	// can be safely used from the inside.
	kj::Thread thread;
	
	// Own<const kj::Executor> executor;
	Own<CrossThreadPromiseFulfiller<void>> doneFulfiller;
		
	InProcessServerImpl(kj::Function<capnp::Capability::Client()> factory, Library library) :
		library(mv(library)),
		factory(mv(factory)),
		
		vatHub(),
		
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
	
	Own<const InProcessServer> addRef() const override { return kj::atomicAddRef(*this); }
	
	capnp::Capability::Client createFor(VatId::Reader clientId) {
		return factory();
	}
	
	void run() {
		// Initialize event loop
		Library library = this->library->addRef();
		auto lt = library -> newThread();
		auto& ws = lt -> waitScope();
		
		// Create server
		using capnp::RpcSystem;
		using fsc::lvn::VatId;
		
		// Move vat network into local scope and shadow it
		Own<LocalVatNetwork> vatNetwork = vatHub.join();
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
		
		// Clear factory to remove cached stuff
		factory = []() -> Service::Client { return nullptr; };
	}
	
	LocalVatHub getHub() const override {
		return vatHub;
	}
};

} // Anonymous namespace

capnp::Capability::Client connectInProcess(const LocalVatHub& hub, uint64_t address) {
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
	
	using capnp::RpcSystem;
	
	Shared<LocalVatNetwork> vatNetwork = hub.join();
	Shared<capnp::RpcSystem<lvn::VatId>> rpcClient(*vatNetwork, nullptr);
	
	Temporary<lvn::VatId> vatId;
	vatId.setKey(address);
	
	auto client = rpcClient -> bootstrap(vatId);
	
	Own<void> attachments = kj::attachRef(client, vatNetwork, rpcClient);
	Promise<void> lifetimeScope = getActiveThread().lifetimeScope().wrap(Promise<void>(NEVER_DONE)).attach(mv(attachments));
	return capnp::membrane(mv(client), kj::refcounted<KeepaliveMembrane>(mv(lifetimeScope)));
}

Own<const InProcessServer> newInProcessServer(kj::Function<capnp::Capability::Client()> serviceFactory, Library lib) {
	return kj::atomicRefcounted<InProcessServerImpl>(mv(serviceFactory), mv(lib));
}

}
