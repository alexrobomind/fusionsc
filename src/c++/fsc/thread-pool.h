#pragma once

#include "common.h"
#include "local.h"

namespace fsc {
	
struct LightWorkerThread {
	LightWorkerThread();
	~LightWorkerThread();
	const kj::Executor& getExecutor();
	
private:
	void run();
	
	kj::MutexGuarded<Maybe<Own<const kj::Executor>>> executor;
	Own<kj::CrossThreadPromiseFulfiller<void>> onDestroy;
	
	kj::Thread thread;
};

struct LightWorkerPool {
	LightWorkerPool(size_t numWorkers = 0);
	
	const kj::Executor& select();
	
	kj::Array<Own<LightWorkerThread>> workers;

private:
	size_t offset = 0;
	size_t base = 0;
};

}