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
	shutdownMode(false),
	stewardThread([this, elevated = params.elevated]() { runSteward(elevated); })
{
	if(params.elevated) {
		KJ_REQUIRE(elevatedInstance == nullptr, "Can only have one active elevated instance");
		elevatedInstance = this;
		
		#ifndef _WIN32
		kj::UnixEventPort::captureChildExit();
		#endif
	}
	
	KJ_IF_MAYBE(pStore, params.dataStore) {
		sharedStore = mv(*pStore);
	} else {
		sharedStore = createStore();
	}
	
	stewardThread.detach();
	
	// Wait for steward thread to finish starting.
	steward();
};

LibraryHandle::~LibraryHandle() {
	stopSteward();
	
	if(elevatedInstance == this)
		elevatedInstance = nullptr;
}

void LibraryHandle::stopSteward() const {
	stewardFulfiller -> fulfill(inShutdownMode());
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

void LibraryHandle::runSteward(bool elevated) {
	#ifndef _WIN32
	if(elevated)
		kj::UnixEventPort::captureChildExit();
	#endif

	// We use a special thread context for the steward that doesn't
	// have back-access to the library.
	StewardContext ctx;
	
	// Register fulfiller for shutdown	
	auto paf = kj::newPromiseAndCrossThreadFulfiller<bool>();
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
	bool fastShutdown = runPromise.wait(ctx.waitScope());
	
	runningLoop = READY_NOW;
	
	if(fastShutdown)
		ctx.shutdownFast();
	
	// DO NOT USE this PAST THIS POINT
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

	// We need to turn the event loop so that we can make sure the cancellations
	// have propagated.
	waitScope().poll();
	
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
	_dataService(kj::heap<LocalDataService>(l)),
	refData(new kj::MutexGuarded<RefData>())
{}

ThreadHandle::~ThreadHandle() {	
	// If the library is in shutdown mode, we have to (conservatively) assume 
	// that all other threads  have unexpectedly died and this is the last
	// remaining active thread. We can not BANK on it, but we can not wait
	// for any other promises on the event loop to resolve.
	if(!_library->inShutdownMode()) {
		drain().wait(waitScope());
		
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
		
		delete refData;
	} else {
		detachedTasks.clear();
		
		bool canDeleteRefdata = false;
		{
			auto locked = refData -> lockExclusive();
		
			if(locked -> refs.size() == 0)
				canDeleteRefdata = true;
		}
		
		if(canDeleteRefdata)
			delete refData;
		
		// Tell the thread context that we are doing a fast shutdown
		ThreadContext::fastShutdown = true;
	}
}

Own<ThreadHandle> ThreadHandle::addRef() {
	return Own<ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}

Own<const ThreadHandle> ThreadHandle::addRef() const {
	return Own<const ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}

// === class StewardContext ===

StewardContext::StewardContext() {};
StewardContext::~StewardContext() {
	detachedTasks.clear();
};

void StewardContext::shutdownFast() {
	ThreadContext::fastShutdown = true;
}

}