#include <list>

#include "local.h"
#include "data.h"
#include "streams.h"

#ifndef _WIN32
#include <kj/async-unix.h>
#endif

namespace fsc {
	
// === class LibraryHandle ===
	
LibraryHandle::LibraryHandle(StartupParameters params) :
	stewardThread([this]() { runSteward(); })
{	
	KJ_IF_MAYBE(pStore, params.dataStore) {
		sharedStore = mv(*pStore);
	} else {
		sharedStore = createStore();
	}
	
	// stewardThread.detach();
	
	// Wait for steward thread to finish starting.
	steward();
};

LibraryHandle::~LibraryHandle() {
	stopSteward();
}

void LibraryHandle::stopSteward() const {
	stewardFulfiller -> fulfill();
}

std::unique_ptr<Botan::HashFunction> LibraryHandle::defaultHash() const {
	auto result = Botan::HashFunction::create("Blake2b");
	KJ_REQUIRE(result != nullptr, "Requested hash function not available");
	return result;
}

const kj::Executor& LibraryHandle::steward() const {
	auto locked = stewardExecutor.lockExclusive();
	locked.wait([](const Maybe<Own<const kj::Executor>>& exec) { return exec != nullptr; });
	
	FSC_ASSERT_MAYBE(pExecutor, *locked, "Internal error");
		
	// The life of this is guarded by the lifetime of the LibraryHandle instance,
	// so this is safe.
	return **pExecutor;
}

void LibraryHandle::runSteward() {
	// We use a special thread context for the steward that doesn't
	// have back-access to the library.
	StewardContext ctx;
	
	// Register fulfiller for shutdown	
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	stewardFulfiller = mv(paf.fulfiller);
	
	// Pass back the executor
	{
		auto locked = stewardExecutor.lockExclusive();
		*locked = ctx.executor().addRef();
	}

	auto runPromise = mv(paf.promise);
	
	// Execute GC loop
	kj::Function<Promise<void>()> gcLoop = [this, &ctx, &gcLoop]() {
		store().gc();
		
		return ctx.timer().afterDelay(1 * kj::MINUTES)
		.then(gcLoop);
	};
	
	Promise<void> runningLoop = gcLoop().eagerlyEvaluate([](kj::Exception&& e) {
		KJ_LOG(WARNING, "Store GC failure", e);
	});

	// loopbackReferenceForStewardStartup = nullptr;
	runPromise.wait(ctx.waitScope());
}

// === class NullErrorHandler ===

void NullErrorHandler::taskFailed(kj::Exception&& e) {
}

NullErrorHandler NullErrorHandler::instance;

// === class ThreadContext ===

ThreadContext::ThreadContext(Maybe<kj::EventPort&> port) :
	asyncInfrastructure(makeAsyncInfrastructure(port)),
	_executor(kj::getCurrentThreadExecutor()),
	_filesystem(kj::newDiskFilesystem()),
	_streamConverter(newStreamConverter()),
	detachedTasks(NullErrorHandler::instance)
{	
	KJ_REQUIRE(current == nullptr, "Can only have one active thread context per thread");
	current = this;
}

ThreadContext::~ThreadContext() {
	KJ_REQUIRE(current == this, "Destroying LibraryThread in wrong thread");
	
	scopeProvider.cancel("Thread context destroyed");
	waitScope().cancelAllDetached();

	// We need to turn the event loop so that we can make sure the cancellations
	// have propagated.
	waitScope().poll();
	
	drain().wait(waitScope());
	
	current = nullptr;
}

OneOf<kj::AsyncIoContext, ThreadContext::CustomEventPort> ThreadContext::makeAsyncInfrastructure(Maybe<kj::EventPort&> port) {
	KJ_IF_MAYBE(pPort, port) {
		return CustomEventPort(*pPort);
	} else {
		return kj::setupAsyncIo();
	}
}


void ThreadContext::detach(Promise<void> p) {
	detachedTasks.add(mv(p));
}

Promise<void> ThreadContext::drain() {
	return detachedTasks.onEmpty();
}

Promise<void> ThreadContext::uncancelable(Promise<void> p) {
	auto forked = p.fork();
	detach(forked.addBranch());
	return forked.addBranch();
}

kj::Canceler& ThreadContext::lifetimeScope() { return scopeProvider; }

ThreadContext::CustomEventPort::CustomEventPort(kj::EventPort& port) :
	loop(kj::heap<kj::EventLoop>(port)),
	waitScope(kj::heap<kj::WaitScope>(*loop))
{}
	
// === class ThreadHandle ===

struct ThreadHandle::Ref {		
	Ref(const kj::MutexGuarded<RefData>* refData);
	~Ref();
		
	kj::ListLink<Ref> link;
	const kj::MutexGuarded<RefData>* refData;
};

struct ThreadHandle::RefData {
	kj::List<Ref, &Ref::link> refs;
	Own<CrossThreadPromiseFulfiller<void>> whenNoRefs;
	const ThreadHandle* owner;
};
		
ThreadHandle::Ref::Ref(const kj::MutexGuarded<RefData>* refData) :
	refData(refData)
{
	refData -> lockExclusive() -> refs.add(*this);
}

ThreadHandle::Ref::~Ref() {
	auto locked = refData -> lockExclusive();
	locked -> refs.remove(*this);
	
	auto& pFulfiller = locked -> whenNoRefs;
	if(pFulfiller.get() != nullptr) {
		pFulfiller->fulfill();
		pFulfiller = nullptr;
	}
}

ThreadHandle::ThreadHandle(Library l, Maybe<kj::EventPort&> eventPort) :
	ThreadContext(eventPort),
	_library(l -> addRef()),
	_dataService(kj::heap<LocalDataService>(l))
	// refData(new kj::MutexGuarded<RefData>())
{}

ThreadHandle::~ThreadHandle() {		
	/*KJ_DBG("Waiting for refs to release");
	while(true) {
		Promise<void> noMoreRefs = READY_NOW;
		
		{
			auto locked = refData -> lockExclusive();
			
			if(locked -> refs.size() == 0)
				break;
			
			auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
			noMoreRefs = mv(paf.promise);
			
			locked -> whenNoRefs = mv(paf.fulfiller);
		}
		
		noMoreRefs.wait(waitScope());
	}
	KJ_DBG("Refs released");*/
	
	// delete refData;
}

/*Own<ThreadHandle> ThreadHandle::addRef() {
	return Own<ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}

Own<const ThreadHandle> ThreadHandle::addRef() const {
	return Own<const ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}*/

// === class StewardContext ===

StewardContext::StewardContext() {};
StewardContext::~StewardContext() {
	// detachedTasks.clear();
};


}