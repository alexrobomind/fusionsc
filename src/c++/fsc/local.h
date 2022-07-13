#pragma once

#include <kj/memory.h>
#include <kj/async-io.h>
#include <kj/debug.h>
#include <kj/refcount.h>
#include <kj/filesystem.h>

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

struct DaemonRunner : public kj::AtomicRefcounted {
	DaemonRunner(const kj::Executor& target);
	inline DaemonRunner() : DaemonRunner(kj::getCurrentThreadExecutor()) {}
	
	inline ~DaemonRunner() { disconnect(); }
	
	inline Own<const DaemonRunner> addRef() const { return kj::atomicAddRef(*this); }
	
	/**
	 * Runs the given task in the event loop associated with this runner, if it is still available.
	 * If the runner is disconnected, this function returns false.
	 *
	 * \returns true if the task could be scheduled to the target event loop, false if the runner
	 *          is disconnected.
	 */
	bool run(kj::Function<Promise<void>()> func) const;
		
	/**
	 * Disconnects the runner from the target thread. After returning, the target event loop can be
	 * destroyed and existing references to this runner can safely outlive it. All attempts to schedule
	 * tasks will fail by returning false, and the passed functions will be destroyed immediately.
	 */
	void disconnect() const;
	
	/**
	 * Blocks the current thread until all daemon tasks have finished.
	 * \note Since usually daemon tasks run forever or until cancelled,
	 *       this method is intended mostly for debug purposes.
	 */
	Promise<void> whenDone() const;

private:
	struct Connection {
		Own<const kj::Executor> executor;
		Own<kj::TaskSet> tasks;
	};
	
	kj::MutexGuarded<Maybe<Connection>> connection;
};

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
	inline void setShutdownMode() { *(shutdownMode.lockExclusive()) = true; }
	
private:
	inline LibraryHandle() :
		storeSteward(store),
		_daemonRunner(kj::atomicRefcounted<DaemonRunner>(storeSteward.getExecutor())),
		shutdownMode(false)
	{};
	
	inline ~LibraryHandle() {
		_daemonRunner->disconnect();
	}
	
	mutable LocalDataStore::Steward storeSteward;
	Own<DaemonRunner> _daemonRunner;
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
};

inline ThreadHandle& getActiveThread() {
	KJ_REQUIRE(ThreadHandle::current != nullptr, "No active thread");
	return *ThreadHandle::current;
}

inline bool hasActiveThread() {
	return ThreadHandle::current != nullptr;
}


/**
 * Class that can be used to connect two different threads with each other.
 * Must be constructed in the context of a running event loop. When both accept
 * and connect are called, their promises resolve to two ends of a local connection.
 * Each method can be called by any thread with an active event loop. Currently,
 * the local connection is implemented by a loopback network TCP/IP connection.
 *
 * The information required to connect to the other side (currently, the
 * accepting port number) is exchanged through the event loop present during the
 * constructor of CrossThreadConnection. This event loop must run until the port
 * information is exchanged, otherwise the promises will not resolve.
 *
 * accept() and connect() can be called in any order, just as their promises
 * can be waited on in any order. They can also be called from a single thread,
 * as long as this thread (obviously) doesn't wait() on any of the two promises
 * before having both methods called.
 *
 * WARNING: Currently, this object must be kept alive until the handshake is complete.
 * Preferably, this restriction will be lifted in the future.
 */
class CrossThreadConnection {
public:
	CrossThreadConnection();
	
	Promise<Own<kj::AsyncIoStream>> accept(ThreadHandle& h);
	Promise<Own<kj::AsyncIoStream>> connect(ThreadHandle& h);
	
	// Size of the random security token transfered to validate the connection
	static constexpr size_t tokenSize = 128;
	
private:
	const kj::Executor& _exec;
	kj::PromiseFulfillerPair<kj::uint> _port;
	
	bool _acceptCalled;
	bool _connectCalled;
	
	FixedArray<byte, tokenSize> sendToken1;
	FixedArray<byte, tokenSize> recvToken1;
	
	FixedArray<byte, tokenSize> sendToken2;
	FixedArray<byte, tokenSize> recvToken2;
};

kj::TwoWayPipe newPipe();

template<typename Func>
kj::Maybe<kj::Promise<UnwrapIfPromise<UnwrapMaybe<ReturnType<Func>>>>> executeMaybe(const kj::Executor& executor, Func&& func) {
	using T = UnwrapIfPromise<UnwrapMaybe<decltype(func())>>;

	// Run f on the other side to get the maybe
	kj::Maybe<kj::Promise<T>> maybe = executor.executeSync(kj::fwd<Func>(func));
	
	KJ_IF_MAYBE(ptr, maybe) {
		// If the maybe is not empty, use the executor to transfer the contained promise
		return executor.executeAsync([&]() { return kj::mv(*ptr); });
	} else {
		return nullptr;
	}
}

// Inline implementations

LibraryThread LibraryHandle::newThread() const {
	return kj::heap<ThreadHandle>(addRef());
}

} // namespace fsc