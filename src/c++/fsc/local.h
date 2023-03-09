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

namespace fsc {
	class LocalDataService;
}

namespace fsc {
	
class LibraryHandle;
using Library = Own<const LibraryHandle>;

class ThreadHandle;
using LibraryThread = Own<ThreadHandle>;

/**
 *  "Global" libary handle. This class serves as the dependency injection context
 *  for objects that should be shared across all threads. Currently, this is only
 *  the local data store table and a shared daemon runner.
 */
class LibraryHandle : public kj::AtomicRefcounted {
public:
	// Mutex protected local data store
	kj::MutexGuarded<LocalDataStore> store;
	
	// Creates an additional owning reference to this handle.
	inline kj::Own<const LibraryHandle> addRef() const { return kj::atomicAddRef(*this); }
	
	// TODO: Currently the daemon runner uses the event loop of the steward. Refactor
	//       the steward to be a daemon instead ...
	
	inline LibraryThread newThread() const ;
	void stopSteward() const;
	
	inline const DaemonRunner& daemonRunner() const { return *_daemonRunner; }
	
	inline bool inShutdownMode() const { return *(shutdownMode.lockShared()); }
	inline void setShutdownMode() const { *(shutdownMode.lockExclusive()) = true; }
	
	std::unique_ptr<Botan::HashFunction> defaultHash() const;
	
private:
	inline LibraryHandle() :
		storeSteward(store),
		shutdownMode(false)
	{};
	
	inline ~LibraryHandle() {
	}
	
	mutable LocalDataStore::Steward storeSteward;
	kj::MutexGuarded<bool> shutdownMode;
	
	friend Own<LibraryHandle> kj::atomicRefcounted<LibraryHandle>();
};


inline Library newLibrary() {
	return kj::atomicRefcounted<LibraryHandle>();
}

/**
 * Thread-local library handle. This holds local infrastructure components required by the
 * library, but which may not be shared in-between threads.
 */
class ThreadHandle /*: public kj::Refcounted*/ {
public:
	// Creates a new library handle from a library handle
	ThreadHandle(Library lh);
	~ThreadHandle();
	
	// Accessors for local use only
	
	// Convenience method to retrieve the wait scope from the io context
	inline kj::WaitScope&      waitScope() { return _ioContext.waitScope; }
	
	/**
	 * IO context. This object holds the event loop, its wait scope, and IO providers to
	 * access network and the file system.
	 */
	inline kj::AsyncIoContext& ioContext() { return _ioContext; }
	
	/**
	 * System clock timer
	 */
	inline kj::Timer& timer() { return _ioContext.provider->getTimer(); }
		
	/**
	 * This thread's random number generator.
	 */
	inline CSPRNG&             rng()       { return _rng; }
	
	/**
	 * Network interface routines
	 */
	inline kj::Network&        network()   { return _ioContext.provider -> getNetwork(); }
	
	kj::Array<const byte> randomID() {
		auto result = kj::heapArray<byte>(16);
		rng().randomize(result);
		return result;
	}
	
	LocalDataService& dataService() { return *_dataService; }
	
	kj::Filesystem& filesystem() { return *_filesystem; }
	
	// Accessors that may be used cross-thread
	
	// Convenience method to retrieve the local data store from the library
	inline const Library&                          library()      const { return _library; }
	inline const kj::MutexGuarded<LocalDataStore>& store()        const { return _library -> store; }
	inline const kj::Executor&                     executor()     const { return _executor; }
	inline const DaemonRunner&                     daemonRunner() const { return _library -> daemonRunner(); }
	
	//! Creates a keep-alive reference to this thread
	/**
	 * Creates a reference that acts similar to a shared pointer. However, instead of shared ownership
	 * semantics, holding this reference causes the ThreadHandle's destructor to cycle its event
	 * loop until all references are destroyed.
	 */
	Own<ThreadHandle> addRef();
	Own<const ThreadHandle> addRef() const;
	
	template<typename T>
	Promise<T> uncancelable(Promise<T> p);
	
	Promise<void> uncancelable(Promise<void> p);
	
	void detach(Promise<void> p);
	Promise<void> drain();
		
private:
	kj::AsyncIoContext _ioContext;
	CSPRNG _rng;
	
	// Back-reference to the library handle.
	Library _library;
	
	const kj::Executor& _executor;
	
	Own<LocalDataService> _dataService;
	Own<kj::Filesystem> _filesystem;
	
	// Access to currenly active instance
	inline static thread_local ThreadHandle* current = nullptr;
	friend ThreadHandle& getActiveThread();
	friend bool hasActiveThread();
	
	struct Ref;
	struct RefData;
	
	kj::MutexGuarded<RefData>* refData;
	
	kj::TaskSet detachedTasks;
};

inline ThreadHandle& getActiveThread() {
	KJ_REQUIRE(ThreadHandle::current != nullptr, "No active thread");
	return *ThreadHandle::current;
}

inline bool hasActiveThread() {
	return ThreadHandle::current != nullptr;
}

// Inline implementations

template<typename T>
Promise<T> LibraryThread::uncancelable(Promise<T> p) {
	auto promiseTuple = p.then([](T t) {
		return kj::tuple(t, 0);
	).split();
	
	detach(kj::get<1>(promiseTuple).ignoreResult());
	return mv(kj::get<0>(promiseTuple));
}

LibraryThread LibraryHandle::newThread() const {
	return kj::heap<ThreadHandle>(addRef());
}

} // namespace fsc