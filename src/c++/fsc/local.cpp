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
	sharedStore(createStoreFromSettings(params)),
	workerPool(*this, params.numWorkerThreads)
{		
	// Start steward thread
	worker().executeSync([this]() {
		runSteward();
	});
};

LibraryHandle::~LibraryHandle() {}

DataStore LibraryHandle::createStoreFromSettings(StartupParameters& params) {
	KJ_IF_MAYBE(pStore, params.dataStore) {
		return *pStore;
	} else {
		return createStore();
	}
}

std::unique_ptr<Botan::HashFunction> LibraryHandle::defaultHash() const {
	auto result = Botan::HashFunction::create("Blake2b");
	KJ_REQUIRE(result != nullptr, "Requested hash function not available");
	return result;
}

void LibraryHandle::runSteward() {
	store().gc();
	
	// Run steward every minute
	auto& at = getActiveThread();
	return at.detach(
		at.timer().afterDelay(1 * kj::MINUTES)
		.then([this]() { runSteward(); })
	);
}

const kj::Executor& LibraryHandle::worker() const {
	return workerPool.select();
}

// === class NullErrorHandler ===

void NullErrorHandler::taskFailed(kj::Exception&& e) {
}

NullErrorHandler NullErrorHandler::instance;

// === class ThreadContext ===

ThreadContext::ThreadContext(Library lh, Maybe<kj::EventPort&> port) :
	_library(mv(lh)),
	asyncInfrastructure(makeAsyncInfrastructure(port)),
	_executor(kj::getCurrentThreadExecutor()),
	_filesystem(kj::newDiskFilesystem()),
	_streamConverter(newStreamConverter()),
	_dataService(kj::heap<LocalDataService>(*_library)),
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

// === class WorkerContext ===

WorkerContext::WorkerContext(Library l) :
	ThreadContext(mv(l))
{}

WorkerContext::~WorkerContext() {
	auto& ws = waitScope();
	ws.cancelAllDetached();
	
	while(!detachedTasks.isEmpty()) {
		detachedTasks.clear();
		ws.cancelAllDetached();
	}
};

}