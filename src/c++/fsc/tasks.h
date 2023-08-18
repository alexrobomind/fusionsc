#include <kj/async.h>

namespace fsc {

struct Task : kj::Canceler {
	virtual void begin(kj::StringPtr, uint64_t) = 0;
	virtual void setStatus(kj::StringPtr) = 0;
	virtual void progress(uint64_t) = 0;
	virtual void done() = 0;
	
	virtual Promise<void> onReject();
	virtual Own<Task> createSubtask(uint64_t) = 0;
}

struct SubtaskBase : public Task {	
	uint64_t budgetInParent = 0;
	uint64_t usedInParent = 0;
	
	uint64_t base
	
	uint64_t localBudget = 0;
	uint64_t localProgress = 0;
	
	SubtaskBase(uint64_t budgetInParent) : budgetInParent(budgetInParent) {}
	
	void begin(uint64_t total) {
		
}