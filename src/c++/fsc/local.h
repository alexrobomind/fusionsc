#pragma once

#include <kj/memory.h>
#include <kj/async-io.h>
#include <kj/debug.h>

#include <utility>

#include "common.h"
#include "store.h"
#include "random.h"

namespace fsc {

/**
 *  "Global" libary handle. This class serves as the dependency injection context
 *  for objects that should be shared across all threads. Currently, this is only
 *  the local data store table.
 */
class Library : public kj::AtomicRefcounted {
public:
	// Mutex protected local data store
	kj::MutexGuarded<LocalDataStore> store;
	
	// Creates an additional owning reference to this handle.
	inline kj::Own<const Library> addRef() const { return kj::atomicAddRef(*this); }
	
	// Creates a refcounted instance of this class
	static inline kj::Own<const Library> create() { return kj::atomicRefcounted<Library>(); }
	
private:
	inline Library() {};
	
	friend kj::Own<Library> kj::atomicRefcounted<Library>();
};

/**
 * Thread-local library handle. This holds local infrastructure components required by the
 * library, but which may not be shared in-between threads.
 */
class ThreadHandle : public kj::Refcounted {
public:
	// Creates a new library handle from a library handle
	ThreadHandle(const Library* l);
	
	// Accessors for local use only
	
	// Convenience method to retrieve the wait scope from the io context
	inline kj::WaitScope&      waitScope() { return _ioContext.waitScope; }
	
	/**
	 * IO context. This object holds the event loop, its wait scope, and IO providers to
	 * access network and the file system.
	 */
	inline kj::AsyncIoContext& ioContext() { return _ioContext; }
	
	/**
	 * This thread's random number generator.
	 */
	inline CSPRNG&             rng()       { return _rng; }
	
	/**
	 * Network interface routines
	 */
	inline kj::Network&        network()   { return _ioContext.provider -> getNetwork(); }
	
	// Accessors that may be used cross-thread
	
	// Convenience method to retrieve the local data store from the library
	inline const kj::Own<const Library>&           library()  const { return _library; }
	inline const kj::MutexGuarded<LocalDataStore>& store()    const { return _library -> store; }
	inline const kj::Executor&                     executor() const { return _executor; }
	
	// Obtain an additional reference. Requres this object to be acquired through create()
	inline kj::Own<ThreadHandle> addRef() { return kj::addRef(*this); }
	
	// Creates a new thread handle on the current thread
	static inline kj::Own<ThreadHandle> create(const Library* l) { return kj::refcounted<ThreadHandle>(l); }
	
	
private:
	kj::AsyncIoContext _ioContext;
	CSPRNG _rng;
	
	// Back-reference to the library handle.
	kj::Own<const Library> _library;
	
	const kj::Executor& _executor;
};

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
	
	// 
	Promise<Own<kj::AsyncIoStream>> accept(ThreadHandle& h);
	Promise<Own<kj::AsyncIoStream>> connect(ThreadHandle& h);
	
	// Size of the random security token transfered to validate the connection
	static constexpr size_t tokenSize = 128;
	
private:
	const kj::Executor& _exec;
	kj::PromiseFulfillerPair<kj::uint> _port;
	
	bool _acceptCalled;
	bool _connectCalled;
	
	FixedArray<byte, tokenSize> sendToken;
	FixedArray<byte, tokenSize> recvToken;
};

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

} // namespace fsc