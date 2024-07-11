#pragma once

#include <kj/memory.h>
#include <kj/async-io.h>
#include <kj/debug.h>
#include <kj/refcount.h>
#include <kj/filesystem.h>

#include <botan/hash.h>

#include <utility>

#include "common.h"
#include "store.h"
#include "random.h"
#include "streams.h"

#include "thread-pool.h"

namespace fsc {
	class LocalDataService;
}

namespace fsc {
	
class LibraryHandle;
using Library = Own<const LibraryHandle>;

class ThreadContext;
using LibraryThread = Own<ThreadContext>;

struct NullErrorHandler : public kj::TaskSet::ErrorHandler {
	static NullErrorHandler instance;
	void taskFailed(kj::Exception&& e) override ;
};

struct StartupParameters {
	Maybe<DataStore> dataStore;
	
	// 0 means auto-detect
	size_t numWorkerThreads = 0;
};

/**
 *  "Global" libary handle. This class serves as the dependency injection context
 *  for objects that should be shared across all threads. Currently, this is only
 *  the local data store table and a shared daemon runner.
 */
struct LibraryHandle : public kj::AtomicRefcounted {
	LibraryHandle(StartupParameters params = StartupParameters());
	~LibraryHandle();
	
	// Creates an additional owning reference to this handle.
	inline kj::Own<const LibraryHandle> addRef() const { return kj::atomicAddRef(*this); }
	
	// Mutex protected local data store
	DataStore& store() const { return sharedStore; }
	
	const kj::Executor& worker() const;
	
	inline LibraryThread newThread(Maybe<kj::EventPort&> eventPort = nullptr) const ;	
	std::unique_ptr<Botan::HashFunction> defaultHash() const;
	
private:
	
	//! Executes the steward thread
	void runSteward();
	
	static DataStore createStoreFromSettings(StartupParameters& params);
	
	mutable DataStore sharedStore;
	
	internal::LightWorkerPool workerPool;
	
	// This loopback reference will be nulled by he steward once it has finished
	// startup and the fulfiller is initialized. This ensures that the destructor
	// is not called before the steward finishes booting up.
	// Maybe<Own<const LibraryHandle>> loopbackReferenceForStewardStartup;
	
	friend Own<LibraryHandle> kj::atomicRefcounted<LibraryHandle>();
	
	friend class StewardContext;
};


inline Library newLibrary(StartupParameters params = StartupParameters()) {
	return kj::atomicRefcounted<LibraryHandle>(mv(params));
}

struct ThreadContext {
	ThreadContext(Library lh, Maybe<kj::EventPort&> = nullptr);
	~ThreadContext();
	
	//! Access to executor from outside threads
	inline const kj::Executor& executor() const {
		return _executor;
	}
	
	inline const Library& library() const { return _library; }
	
	inline LocalDataService& dataService() {
		return *_dataService;
	}
	inline kj::AsyncIoContext& ioContext() {
		KJ_REQUIRE(asyncInfrastructure.is<kj::AsyncIoContext>(), "Can only perform async IO in a thread with a default event port");
		
		return asyncInfrastructure.get<kj::AsyncIoContext>();
	}
	inline kj::WaitScope& waitScope() {
		if(asyncInfrastructure.is<kj::AsyncIoContext>())
			return asyncInfrastructure.get<kj::AsyncIoContext>().waitScope;
		else
			return *asyncInfrastructure.get<CustomEventPort>().waitScope;
	}
	inline kj::Timer& timer() {
		KJ_REQUIRE(asyncInfrastructure.is<kj::AsyncIoContext>(), "Can only perform timer creation in a thread with a default event port");
		return ioContext().provider->getTimer();
	}
	inline CSPRNG& rng() {
		return _rng;
	}
	inline kj::Network& network() {
		return ioContext().provider -> getNetwork();
	}
	inline StreamConverter& streamConverter() {
		return *_streamConverter;
	}
	inline kj::Filesystem& filesystem() {
		return *_filesystem;
	}
	inline DataStore& store() const {
		return library() -> store();
	}
	inline const kj::Executor& worker() const {
		return library() -> worker();
	}
	
	inline kj::Array<const byte> randomID() {
		auto result = kj::heapArray<byte>(16);
		rng().randomize(result);
		return result;
	}
	
	template<typename T>
	Promise<T> uncancelable(Promise<T> p);
	
	Promise<void> uncancelable(Promise<void> p);
	
	void detach(Promise<void> p);
	Promise<void> drain();
	
	kj::Canceler& lifetimeScope();
	
private:
	struct CustomEventPort {
		Own<kj::EventLoop> loop;
		Own<kj::WaitScope> waitScope;
		
		CustomEventPort(kj::EventPort& port);
	};
	
	// Access to currenly active instance
	inline static thread_local ThreadContext* current = nullptr;
	friend ThreadContext& getActiveThread();
	friend bool hasActiveThread();
	
	OneOf<kj::AsyncIoContext, CustomEventPort> makeAsyncInfrastructure(Maybe<kj::EventPort&> port);
	
	// Fields.
	Library _library;
	OneOf<kj::AsyncIoContext, CustomEventPort> asyncInfrastructure;
	CSPRNG _rng;
	Own<kj::Filesystem> _filesystem;
	Own<StreamConverter> _streamConverter;
	const kj::Executor& _executor;
	Own<LocalDataService> _dataService;
	
	kj::Canceler scopeProvider;

protected:
	kj::TaskSet detachedTasks;
};

inline ThreadContext& getActiveThread() {
	KJ_REQUIRE(ThreadContext::current != nullptr, "No active thread");
	return *ThreadContext::current;
}

inline bool hasActiveThread() {
	return ThreadContext::current != nullptr;
}

//! Thread context that cancels all long-running tasks
struct WorkerContext : public ThreadContext {	
	WorkerContext(Library l);
	~WorkerContext();
};

// Inline implementations

template<typename T>
Promise<T> ThreadContext::uncancelable(Promise<T> p) {
	auto promiseTuple = p.then([](T t) {
		return kj::tuple(t, 0);
	}).split();
	
	detach(kj::get<1>(promiseTuple).ignoreResult());
	return mv(kj::get<0>(promiseTuple));
}

LibraryThread LibraryHandle::newThread(Maybe<kj::EventPort&> eventPort) const {
	return kj::heap<ThreadContext>(addRef(), mv(eventPort));
}

} // namespace fsc