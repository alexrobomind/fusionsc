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

namespace fsc {
	class LocalDataService;
}

namespace fsc {
	
class LibraryHandle;
using Library = Own<const LibraryHandle>;

class ThreadHandle;
using LibraryThread = Own<ThreadHandle>;

struct NullErrorHandler : public kj::TaskSet::ErrorHandler {
	static NullErrorHandler instance;
	void taskFailed(kj::Exception&& e) override ;
};

/**
 *  "Global" libary handle. This class serves as the dependency injection context
 *  for objects that should be shared across all threads. Currently, this is only
 *  the local data store table and a shared daemon runner.
 */
struct LibraryHandle : public kj::AtomicRefcounted {
	LibraryHandle();
	~LibraryHandle();
	
	// Creates an additional owning reference to this handle.
	inline kj::Own<const LibraryHandle> addRef() const { return kj::atomicAddRef(*this); }
	
	// Mutex protected local data store
	inline const kj::MutexGuarded<LocalDataStore>& store() const { return sharedData -> store; }
	
	const kj::Executor& steward() const ;
	
	inline LibraryThread newThread() const ;
	void stopSteward() const;
	
	inline bool inShutdownMode() const { return *(shutdownMode.lockShared()); }
	inline void setShutdownMode() const { *(shutdownMode.lockExclusive()) = true; }
	
	std::unique_ptr<Botan::HashFunction> defaultHash() const;
	
	/**
	 * Sets this FSC library instance to elevated mode. This causes the following changed:
	 *  - On UNIX, installs a signal handler to handle SIGCHLD events. The handler is restricted
	 *    to the event loop of the steward thread. Since SIGCHLD handlers tend to heavily interfere
	 *    with each other, this action is restricted to on-demand.
	 *
	 * \warning You can only have one elevated instance per process.
	 */
	void elevate();
	bool isElevated () { return elevatedInstance == this; }
	
private:
	struct SharedData : public kj::AtomicRefcounted {
		kj::MutexGuarded<LocalDataStore> store;
	};
	
	//! Executes the steward thread
	void runSteward();
	
	kj::MutexGuarded<bool> shutdownMode;
	Own<SharedData> sharedData;
	
	Own<kj::CrossThreadPromiseFulfiller<void>> stewardFulfiller;
	kj::MutexGuarded<Maybe<Own<const kj::Executor>>> stewardExecutor;
	Promise<void> stewardTask;
	kj::Thread stewardThread;
	
	// This loopback reference will be nulled by he steward once it has finished
	// startup and the fulfiller is initialized. This ensures that the destructor
	// is not called before the steward finishes booting up.
	// Maybe<Own<const LibraryHandle>> loopbackReferenceForStewardStartup;
	
	friend Own<LibraryHandle> kj::atomicRefcounted<LibraryHandle>();
	
	static inline LibraryHandle* elevatedInstance = nullptr;
	friend class StewardContext;
};


inline Library newLibrary() {
	return kj::atomicRefcounted<LibraryHandle>();
}

struct ThreadContext {
	ThreadContext();
	~ThreadContext();
	
	//! Access to executor from outside threads
	virtual const Library& library() const = 0;
	
	virtual LocalDataService& dataService() = 0;
	
	inline const kj::Executor& executor() const {
		return _executor;
	}
	inline kj::AsyncIoContext& ioContext() {
		return _ioContext;
	}
	inline kj::WaitScope& waitScope() {
		return ioContext().waitScope;
	}
	inline kj::Timer& timer() {
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
	inline const kj::MutexGuarded<LocalDataStore>& store() const {
		return library() -> store();
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
	
private:
	// Access to currenly active instance
	inline static thread_local ThreadContext* current = nullptr;
	friend ThreadContext& getActiveThread();
	friend bool hasActiveThread();
	
	// Fields
	kj::AsyncIoContext _ioContext;
	CSPRNG _rng;
	Own<kj::Filesystem> _filesystem;
	Own<StreamConverter> _streamConverter;
	const kj::Executor& _executor;

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

/**
 * Thread-local library handle. This holds local infrastructure components required by the
 * library, but which may not be shared in-between threads.
 */
struct ThreadHandle : public ThreadContext {
	// Creates a new thread handle from a library handle
	ThreadHandle(Library lh);
	~ThreadHandle();
	
	LocalDataService& dataService() override { return *_dataService; }
	inline const Library& library() const override { return _library; }
	
	//! Creates a keep-alive reference to this thread
	/**
	 * Creates a reference that acts similar to a shared pointer. However, instead of shared ownership
	 * semantics, holding this reference causes the ThreadHandle's destructor to cycle its event
	 * loop until all references are destroyed.
	 */
	Own<ThreadHandle> addRef();
	Own<const ThreadHandle> addRef() const;
		
private:
	// Back-reference to the library handle.
	Library _library;
	Own<LocalDataService> _dataService;
	
	struct Ref;
	struct RefData;
	
	kj::MutexGuarded<RefData>* refData;
};

struct StewardContext : public ThreadContext {
	inline const Library& library() const override { KJ_FAIL_REQUIRE("Library instance not available from steward context"); }
	inline LocalDataService& dataService() override { KJ_FAIL_REQUIRE("Data service not available from steward context"); }
	
	StewardContext();
	~StewardContext();
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

LibraryThread LibraryHandle::newThread() const {
	return kj::heap<ThreadHandle>(addRef());
}

} // namespace fsc