#include "operation.h"

namespace fsc {
	
Operation::~Operation() {
	fail(KJ_EXCEPTION(FAILED, "Operation cancelled"));
	clear();
}

Own<const Operation> Operation::addRef() const {
	return kj::atomicAddRef(*this);
}

Own<Operation> Operation::newChild() const {
	auto result = newOperation();
	result -> attachDestroyAnywhere(this->addRef());
	return result;
}

void Operation::dependsOn(Promise<void> promise) const {
	// Registering the dependency requires connection of two different objects
	// A dependency node on the Op side that also owns the promise, and a failure
	// propagator. Because the objects might die in distinct threads, their lifetimes
	// need to be decoupled. Maintaining a cross-link between these different objects
	// requires a third struct that holds the link between the two under a mutex.
	
	// A double-deletion of the link won't happen, because it is only locked exclusively
	// twice. Only the second lock will see both node and propagator as nullptr, and therefore
	// delete the link.
	
	struct DependencyNode;
	struct FailPropagator;
	
	struct Link {
		struct Data {
			DependencyNode* node = nullptr;
			FailPropagator* propagator = nullptr;
		
			bool shouldDelete() {
				return node == nullptr && propagator == nullptr;
			}
		};
		
		kj::MutexGuarded<Data> data;
	};
	
	struct DependencyNode : public Node {
		Own<Promise<void>> dependency;
		const Operation& owner;
		Own<const kj::Executor> deleteExecutor;
		
		DependencyNode(Promise<void> dependency, const Operation& owner) :
			dependency(kj::heap(mv(dependency))),
			owner(owner),
			deleteExecutor(getActiveThread().executor().addRef())
		{}
		
		// Only accessed in clear()
		Link* propagatorLink = nullptr;
		
		void onFinish() override {
			clear();
		}
		void onFailure(kj::Exception&& e) override {
			clear();
		}
		
		Promise<void> destroy() override {
			return deleteExecutor -> executeAsync(
				[ownPromise = mv(dependency)]() mutable {
					// Destroys promise
					ownPromise = nullptr;
				}
			);
			
			// return result.attach(mv(owningThread));
		}
		
		// Clear only gets called once, so this is guaranteed to be safe
		void clear() {
			bool deleteLink = false;
			{
				auto locked = propagatorLink -> data.lockExclusive();
				locked -> node = nullptr;
				
				deleteLink = locked -> shouldDelete();
			}
			
			if(deleteLink) delete propagatorLink;
		}
	};
	
	struct FailPropagator {
		Link* link = nullptr;
		
		~FailPropagator() {
			bool deleteLink = false;
			{
				auto locked = link -> data.lockExclusive();
				locked -> propagator = nullptr;
				
				deleteLink = locked -> shouldDelete();
			}
			
			if(deleteLink) delete link;
		}
		
		void propagate(kj::Exception&& e) {
			// Calling the owning operation's fail() method will unlink that
			// side of the link. This enters the link's mutex, which we also need
			// to get the op.
			// Therefore, we need to acquire a reference to the op first (to prevent
			// its deletion) and then call fail() outside the lock.
			Own<const Operation> op;
			
			{
				auto locked = link -> data.lockShared();
			
				if(locked -> node == nullptr)
					return;
				
				op = locked -> node -> owner.addRef();
			}
			
			op -> fail(mv(e));
		}
	};
	
	// Create failure propagator and attach it to the promise
	auto propagator = heapHeld<FailPropagator>();
	promise = promise
		.eagerlyEvaluate([propagator](kj::Exception&& e) mutable {
			propagator->propagate(mv(e));
		})
		.attach(propagator.x())
		.eagerlyEvaluate(nullptr);
	;
	
	// Create dependency node
	auto node = new DependencyNode(mv(promise), *this);
	
	// Link propagator and dependency node
	auto link = new Link();
	{
		auto locked = link -> data.lockExclusive();
		locked -> node = node;
		locked -> propagator = propagator.get();
	}
	propagator -> link = link;
	node -> propagatorLink = link;
	
	attachNode(node);
}

Promise<void> Operation::whenDone() const {
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	
	struct PromiseNode : public Node {
		kj::Own<kj::CrossThreadPromiseFulfiller<void>> fulfiller;
		
		PromiseNode(kj::Own<kj::CrossThreadPromiseFulfiller<void>>&& fulfiller) :
			fulfiller(mv(fulfiller))
		{}
		
		void onFinish() override {
			fulfiller->fulfill();
		}
		
		void onFailure(kj::Exception&& e) override {
			fulfiller->reject(mv(e));
		}
		
		~PromiseNode() noexcept {}
	};
	
	attachNode(new PromiseNode(mv(paf.fulfiller)));		
	return paf.promise.attach(this->addRef());
}
	
void Operation::attachNode(Node* node) const {
	auto locked = data.lockExclusive();
	
	locked -> nodes.addFront(*node);
	
	if(locked -> state == SUCCESS) {
		node -> onFinish();
	} else if(locked -> state == ERROR) {
		node -> onFailure(cp(* locked -> exception));
	}
}

void Operation::done() const {
	auto locked = data.lockExclusive();
	
	if(locked -> state != ACTIVE)
		return;
	
	locked -> state = SUCCESS;			
	for(Node& node : locked->nodes) {
		node.onFinish();
	}
}

void Operation::fail(kj::Exception&& e) const {
	auto locked = data.lockExclusive();
	
	if(locked -> state != ACTIVE)
		return;
	
	locked -> exception = kj::heap(e);
	locked -> state = ERROR;
	
	for(Node& node : locked->nodes) {
		node.onFailure(cp(e));
	}
}

void Operation::clear() const {
	auto locked = data.lockExclusive();
	
	auto nodes = kj::heapArrayBuilder<Node*>(locked->nodes.size());
	for(auto& node : locked -> nodes) {
		nodes.add(&node);
		locked -> nodes.remove(node);
	}
	
	locked -> clearRunner -> run([nodes = nodes.finish()]() mutable -> Promise<void> {
		Promise<void> result = READY_NOW;
		
		for(Node* pNode : nodes) {
			result = result
			.then([pNode]() { return pNode -> destroy(); })
			.catch_([](kj::Exception e) {})
			.then([pNode]() { delete pNode; });
		}
		
		return result;				
	});
}

Own<Operation> newOperation() {
	auto result = kj::atomicRefcounted<Operation>();
	return result;
}

}