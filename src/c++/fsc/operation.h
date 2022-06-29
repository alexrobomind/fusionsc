#pragma once

#include "local.h"

namespace fsc {

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
	 * promise fails, the operation will fail with the same exception. If it is can-
	 * celled, the operation will continue.
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
	void attachDestroyInThread(const kj::Executor& executor, T&&... params) const ;
	
	/**
	 * Attaches the target objects to live with the operation and be destroyed in this
	 * event loop. See above for details.
	 */
	template<typename... T>
	void attachDestroyHere(T&&... params) const;
	
	/**
	 * Attaches a thread-safe object to be kept during the lifetime of this object. Will
	 * be destroyed in the thread that calls done() / failed() or the last destructor.
	 */
	template<typename... T>
	void attachDestroyAnywhere(T&&... params) const;
	
	Own<Operation> newChild() const;

private:
	struct Node {
		kj::ListLink<Node> link;
		
		virtual void onFinish() {};
		virtual void onFailure(kj::Exception&& e) {};
		virtual ~Node() {};
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
	};
	
	kj::MutexGuarded<Data> data;
	
	void clear() const;
	
	template<typename T>
	void attachNode(T* node) const;
};

Own<Operation> newOperation();

// ================================== Inline implementation =================================

template<typename... T>
void Operation::attachDestroyInThread(const kj::Executor& executor, T&&... params) const {
	struct AttachmentNode : public Node {
		Own<const kj::Executor> executor;
		Own<DaemonRunner> runner;
		
		TupleFor<kj::Decay<T>...> contents;
		
		AttachmentNode(const kj::Executor& executor, T&&... params) :
			executor(executor.addRef()),
			contents(tuple(fwd<T>(params)...))
		{}
		
		void onFinish() override {};
		void onFailure(kj::Exception&& e) override {};
		
		~AttachmentNode() noexcept {
			if(executor.get() != nullptr && runner.get() != nullptr) {
				// Task that schedules the destruction on the actual target thread
				auto destroyTask = [executor = mv(executor), contents = mv(contents)]() mutable -> Promise<void> {
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
			
	attachNode(new AttachmentNode(executor, fwd<T>(params)...));
};

template<typename... T>
void Operation::attachDestroyHere(T&&... params) const {
	this -> attachDestroyInThread(kj::getCurrentThreadExecutor(), fwd<T>(params)...);
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
		~AttachmentNode() noexcept {};
	};
	
	attachNode(new AttachmentNode(fwd<T>(params)...));
}
	
template<typename T>
void Operation::attachNode(T* node) const {
	auto locked = data.lockExclusive();
	
	locked -> nodes.addFront(*node);
	
	if(locked -> state == SUCCESS) {
		node -> onFinish();
	} else if(locked -> state == ERROR) {
		node -> onFailure(cp(* locked -> exception));
	}
}


}