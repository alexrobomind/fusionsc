#pragma once

#include "local.h"

namespace fsc {

/**
 * In several places, FSC calls external asynchronous operations that share resources with Cap'n'Proto
 * This presents two complications:
 * - Completion information has to pass through distinct threads, and the completing thread might not
 *   even have an active event loop.
 * - Associated resources need to be released in owning threads, but potentially in correct order.
 *
 * Operations provide a link between Cap'n'proto's rather strict lifetime requirements and the looser behavior
 * of other libraries. Attached resources are freed in reverse order upon deletion of the last reference to the
 * op. Individual threads can register dependency promises (whose failure will result in failure of the operation
 * ) or acquire the op state as a promise. The API is completely thread-safe, except the usual restriction that
 * promise objects may not be passed between threads (but can be registered or obtained from any potential event loop
 * thread).
 */
struct Operation : kj::AtomicRefcounted {
	~Operation();
	
	Own<const Operation> addRef() const;
	
	/**
	 * Returns a promise that resolves when the operation completes. If the operation
	 * fails or is cancelled, the returned promise will fail.
	 */
	Promise<void> whenDone() const;
	
	/**
	 * Completes all registered and future promises obtained from whenDone()
	 */
	void done() const;
	
	/**
	 * Fails all registered and future promises obtained from whenDone()
	 */
	void fail(kj::Exception&& e) const;
	
	/**
	 * Registers the given promise as a dependency of this operation. If the given
	 * promise fails, the operation will fail with the same exception. Note that this
	 * will only chain the failure state of the op, not cancel whatever work is
	 * associated with this operation. Since operations represent already-running
	 * work, it will be impossible to cancel the passed promise.
	 * The lifetime of the promise is extended to the lifetime of the operation object.
	 * Even if the op fails (which might also indicate a promise the op depends on failing),
	 * the promise will still be kept running.
	 */
	void dependsOn(Promise<void> promise) const;
	
	/**
	 * Attaches the given objects to the promise, so that their lifetime is extended
	 * until all references to this operation are destroyed. If the current library's
	 * daemon runner is still alive, it will try to destroy the attached objects in the
	 * target thread. If that fails, they will be destroyed in the daemon runner thread.
	 * If the daemon runner is already dead, the attached objects will be destroyed in
	 * whatever thread finished / failed / cancelled the operation.
	 */
	template<typename... T>
	void attachDestroyInThread(const ThreadHandle& handle, T&&... params) const ;
	
	/**
	 * Attaches the target objects to live with the operation and be destroyed in this
	 * event loop. See attachDestroyInThread for details.
	 */
	template<typename... T>
	void attachDestroyHere(T&&... params) const;
	
	/**
	 * Attaches a thread-safe object to be kept during the lifetime of this object. Will
	 * be destroyed in the daemon thread.
	 */
	template<typename... T>
	void attachDestroyAnywhere(T&&... params) const;
	
	/**
	 * Creates a new operation that holds a keep-alive reference to the current operation
	 * during its lifetime.
	 */
	Own<Operation> newChild() const;

private:
	inline Operation() {
		auto locked = data.lockExclusive();
		locked -> clearRunner = getActiveThread().daemonRunner().addRef();
	};
	Operation(const Operation& other) = delete;
	Operation(Operation&& other) = delete;
	
	struct Node {
		kj::ListLink<Node> link;
		
		virtual void onFinish() {};
		virtual void onFailure(kj::Exception&& e) {};
		virtual Promise<void> destroy() { return READY_NOW; };
		virtual ~Node() noexcept(false) {};
	};
	
	enum State {
		ACTIVE,
		SUCCESS,
		ERROR
	};
	
	using NodeList = kj::List<Node, &Node::link>;
	
	struct Data {
		State state = ACTIVE;
		NodeList nodes;
		Own<kj::Exception> exception;
		Own<const DaemonRunner> clearRunner;
	};
	
	kj::MutexGuarded<Data> data;
	
	void clear() const;
	
	void attachNode(Node* node) const;
	
	friend Own<Operation> kj::atomicRefcounted<Operation>();
};

Own<Operation> newOperation();

// ================================== Inline implementation =================================

template<typename... T>
void Operation::attachDestroyInThread(const ThreadHandle& thread, T&&... params) const {
	struct AttachmentNode : public Node {
		Own<const ThreadHandle> owningThread;
		
		TupleFor<kj::Decay<T>...> contents;
		
		AttachmentNode(const ThreadHandle& thread, T&&... params) :
			owningThread(thread.addRef()),
			contents(tuple(fwd<T>(params)...))
		{}
		
		void onFinish() override {};
		void onFailure(kj::Exception&& e) override {};
		
		Promise<void> destroy() override {
			auto result = owningThread -> executor().executeAsync(
				[contents = mv(contents)]() mutable {
					// This will move the contents into a local tuple that gets destroyed
					// in the target thread.
					// This is important as the surrounding lambda will get moved back
					// into the calling thread before getting destroyed.
					
					auto destroyedLocally = mv(contents);
				}
			);
			
			return result.attach(mv(owningThread));
		};
	};
			
	attachNode(new AttachmentNode(thread, fwd<T>(params)...));
};

template<typename... T>
void Operation::attachDestroyHere(T&&... params) const {
	this -> attachDestroyInThread(getActiveThread(), fwd<T>(params)...);
}

template<typename... T>
void Operation::attachDestroyAnywhere(T&&... params) const {
	struct AttachmentNode : public Node {			
		TupleFor<kj::Decay<T>...> contents;
		
		AttachmentNode(T&&... params) :
			contents(tuple(fwd<T>(params)...))
		{}
		
		void onFinish() override {};
		void onFailure(kj::Exception&& e) override {};
		Promise<void> destroy() override {
			auto destroyedLocally = mv(contents);
			return READY_NOW;
		}
		
		~AttachmentNode() noexcept {};
	};
	
	attachNode(new AttachmentNode(fwd<T>(params)...));
}


}