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
	// Note: This creates a recursive ownership loop by attaching the op to the promise
	// and vice versa. This is only safe because eventually the incoming promise will eventually
	// resolve (in fact, we expect it to resolve fairly fast), and then the reference to the
	// operation will be deleted.
	promise = promise
		.eagerlyEvaluate([this](kj::Exception&& e) mutable {
			this->fail(cp(e));
		})
		.attach(this->addRef())
		.eagerlyEvaluate(nullptr);
	;
	
	attachDestroyHere(mv(promise));
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