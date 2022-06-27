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

class DaemonRunner : public kj::AtomicRefcounted {
	/**
	 * Runs the given task in the event loop associated with this runner, if it is still available.
	 * If the task can not be run, it will be silently destroyed in the current thread.
	 *
	 * \returns true if the task could be scheduled to the target event loop, false if the target event
	 *          loop is dead
	 */
	bool run(kj::Function<Promise<void>()> func) const;
	
	DaemonRunner(const kj::Executor& target);
	inline DaemonRunner() : DaemonRunner(kj::getCurrentThreadExecutor()) {}
	
	inline Own<const DaemonRunner> addRef() { return kj::atomicAddRef(*this); }
	
	/**
	 * Disconnects the runner from the target thread. After returning, the target event loop can be
	 * destroyed and existing references to this runner can safely outlive it. All attempts to schedule
	 * tasks will fail by returning false, and the passed functions will be destroyed.
	 */
	void disconnect() const;

private:
	struct Connection {
		Own<const kj::Executor> executor;
		Own<kj::TaskSet> taskSet;
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
	
	inline LibraryThread newThread() const ;
	void stopSteward() const;
	
	inline const DaemonRunner& daemonRunner() const { return *_daemonRunner; }
	
private:
	inline LibraryHandle() :
		storeSteward(store),
		_daemonRunner(kj::heap<DaemonRunner>(storeSteward.getExecutor()))
	{};
	
	inline ~LibraryHandle() {
		daemonRunner->disconnect();
	}
	
	mutable LocalDataStore::Steward storeSteward;
	Own<DaemonRunner> _daemonRunner;
	
	friend Own<LibraryHandle> kj::atomicRefcounted<LibraryHandle>();
};


inline Library newLibrary() {
	return kj::atomicRefcounted<LibraryHandle>();
}

/**
 * Thread-local library handle. This holds local infrastructure components required by the
 * library, but which may not be shared in-between threads.
 */
class ThreadHandle : public kj::Refcounted {
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
	inline const Library&                          library()  const { return _library; }
	inline const kj::MutexGuarded<LocalDataStore>& store()    const { return _library -> store; }
	inline const kj::Executor&                     executor() const { return _executor; }
	inline const kj::DaemonRunner&                 daemonRunner() const { return _library -> daemonRunner(); }
	
	// Obtain an additional reference. Requres this object to be acquired through create()
	inline kj::Own<ThreadHandle> addRef() { return kj::addRef(*this); }
	
private:
	kj::AsyncIoContext _ioContext;
	CSPRNG _rng;
	
	// Back-reference to the library handle.
	Library _library;
	
	const kj::Executor& _executor;
	
	Own<LocalDataService> _dataService;
	Own<kj::Filesystem> _filesystem;
	
	inline static thread_local ThreadHandle* current = nullptr;
	
	friend ThreadHandle& getActiveThread();
};

inline ThreadHandle& getActiveThread() {
	KJ_REQUIRE(ThreadHandle::current != nullptr, "No active thread");
	return *ThreadHandle::current;
}

struct Operation : kj::AtomicRefcounted {
	struct Node {
		ListLink<Node> link;
		
		virtual void onFinish() {};
		virtual void onFailure(kj::Exception&& e) {};
		virtual ~Node() {};
	};
	
	using NodeList = kj::List<Node, &node::link>;
	using Data = OneOf<NodeList, kj::Exception, int>;
	kj::MutexGuarded<Data> nodes;
	
	Operation() {
		nodes.lockExclusive() -> init<NodeList>();
	}
	
	~Operation() {
		onFailure(KJ_EXCEPTION("Operation cancelled"));
	}
	
	/**
	 * Registers the given promise as a dependency of this operation. If the given
	 * promise fails, the operation will fail with the same exception.
	 */
	Promise<void> dependsOn(Promise<void> promise) {
		return promise.catch_([this](kj::Exception&& e) {
			this->fail(e);
			kj::throwRecoverableException(e);
		});
	}
	
	/**
	 * Returns a promise that resolves when the operation completes. If the operation
	 * fails or is cancelled, the returned promise will fail.
	 */
	Promise<void> whenDone() const {
		auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
		
		struct PromiseNode {
			kj::Own<CrossThreadFulfiller<void>> fulfiller;
			
			PromiseNode(kj::Own<CrossThreadFulfiller<void>>&& fulfiller) :
				fulfiller(mv(fulfiller))
			{}
			
			void onFinish() override {
				fulfiller->fulfill(kj::READY_NOW);
			}
			
			void onFailure(kj::Exception&& e) override {
				fulfiller->reject(mv(e));
			}
		};
		
		auto locked = nodes.lockExclusive();
		KJ_IF_MAYBE(pNodes, *locked) {
			pNodes -> add(new PromiseNode(mv(paf->fulfiller));
		} else {
			return kj::
		
		return mv(paf.promise);
	}
	
	/**
	 * Attaches the given objects to the promise, so that their lifetime is extended
	 * until the operation completes, fails, or is cancelled. If the current library's
	 * daemon runner is still alive, it will try to destroy the attached objects in the
	 * target thread. If that fails, they will be destroyed in the daemon runner thread.
	 * If the daemon runner is already dead, the attached objects will be destroyed in
	 * whatever thread finished / failed / cancelled the operation.
	 */
	template<typename... T>
	void attachDestroyInThread(const kj::Executor& executor, T&&... params) {
		struct AttachmentNode {
			Own<const kj::Executor> executor;
			Own<DaemonRunner> runner;
			
			kj::TupleFor<kj::Decay<T>...> contents;
			
			AttachmentNode(const Executor& executor, T&&... params) :
				executor(executor.addRef()),
				tuple(fwd<T...>(params))
			{}
			
			void onFinish() override {};
			void onFailure(kj::Exception&& e) override {};
			
			void ~AttachmentNode() {
				if(executor != nullptr && runner != nullptr) {
					// Task that schedules the destruction on the actual target thread
					auto destroyTask = [executor = mv(executor), contents = mv(contents)]() -> Promise<void> mutable {
						return executor->executeAsync([contents = mv(contents)]() mutable {
							// This will move the contents into a local tuple that gets destroyed
							// in the target thread.
							// This is important as the surrounding lambda will get moved back
							// into the calling thread before getting destroyed.
							
							auto destroyedLocally = mv(contents);
						});
					};
					runner -> run(mv(destroyTask));
				}
			}
		};
				
		attachNode(AttachmentNode(executor, fwd<T>(params)...));
	};
	
	/**
	 * Attaches the target objects to live with the operation and be destroyed in this
	 * event loop. See above for details.
	 */
	template<typename... T>
	void attachDestroyHere(T&&... params) const {
		attachDestroyInThread(kj::getCurrentThreadExecutor(), fwd<T>(params)...);
	}
	
	/**
	 * Attaches a thread-safe object to be kept until the end of this operation. Will
	 * be destroyed in the thread that calls done() / failed() or the last destructor.
	 */
	template<typename... T>
	void attachDestroyAnywhere(T&&... params) const {
		struct AttachmentNode {			
			kj::TupleFor<kj::Decay<T>...> contents;
			
			AttachmentNode(T&&... params) :
				tuple(fwd<T...>(params))
			{}
			
			void onFinish() override {};
			void onFailure(kj::Exception&& e) override {};
		};
		
		attachNode(AttachmentNode(fwd<T>(params)...));
	}
	
	void done() const {
		{
			auto locked = nodes.lockExclusive();
			
			if(!locked->is<NodeList>())
				return;
			
			for(Node& node : locked->get<NodeList>()) {
				node.onFinish();
			}
			
			clear(locked->get<NodeList>());
			*locked = (int) 0;
		}
	}
	
	void failed(kj::Exception&& e) const {
		{
			auto locked = nodes.lockExclusive();
			
			if(!locked->is<NodeList>())
				return;
			
			for(Node& node : locked->get<NodeList>()) {
				node.onFailure(cp(e));
			}
			
			clear(locked->get<NodeList>());
			*locked = mv(e);
		}
	}

private:
	void clear(NodeList& nodes) const {
		for(Node& node : *pNodes) {
			locked->remove(node);
			delete &node;
		}
	}
	
	template<typename T>
	void attachNode(T node) {
		auto locked = nodes.lockExclusive();
		if(locked->is<NodeList>()) {
			locked->get<NodeList>().addFront(new AttachmentNode(mv(node)));
		} else if(locked->is<kj::Exception>()) {
			node.onFailure(cp(locked->get<kj::Exception>()));
		} else {
			node.onFinish();
		}
	}
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
	return kj::refcounted<ThreadHandle>(addRef());
}

} // namespace fsc