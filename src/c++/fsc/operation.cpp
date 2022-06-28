#include "operation.h"

namespace fsc {
	
Operation::~Operation() {
	fail(KJ_EXCEPTION(FAILED, "Operation cancelled"));
	clear();
}

Promise<void> Operation::dependsOn(Promise<void> promise) const {
	return promise.catch_([this](kj::Exception&& e) mutable {
		this->fail(cp(e));
		kj::throwRecoverableException(mv(e));
	});
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
	return mv(paf.promise);
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
	
	for(Node& node : locked -> nodes) {
		locked->nodes.remove(node);
		delete &node; // blergh
	}
}

}