#include "thread-pool.h"

#include <thread>

namespace fsc {
	
LightWorkerThread::LightWorkerThread() :
	thread([this]() mutable { run(); })
{
	// Wait until thread is running
	auto locked = executor.lockExclusive();
	locked.wait([](auto& maybeExec) { return maybeExec != nullptr; });
}

LightWorkerThread::~LightWorkerThread() {
	if(onDestroy.get() != nullptr) {
		onDestroy -> fulfill();
	}
}

void LightWorkerThread::run() {
	kj::EventLoop eventLoop;
	kj::WaitScope ws(eventLoop);
	
	// Create cross thread promise
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	onDestroy = mv(paf.fulfiller);
	
	// Signal constructor we are ready
	*(executor.lockExclusive()) = kj::getCurrentThreadExecutor().addRef();
	
	// Wait until destruction
	paf.promise.wait(ws);
	ws.cancelAllDetached();
}

const kj::Executor& LightWorkerThread::getExecutor() {
	auto locked = executor.lockShared();
	
	KJ_IF_MAYBE(pExec, *locked) {
		return **pExec;
	}
	
	KJ_UNREACHABLE;
}

LightWorkerPool::LightWorkerPool(size_t nWorkers) {
	if(nWorkers == 0)
		nWorkers = std::thread::hardware_concurrency();
	
	auto builder = kj::heapArrayBuilder<Own<LightWorkerThread>>(nWorkers);
	
	for(auto i : kj::range(0, nWorkers)) {
		builder.add(kj::heap<LightWorkerThread>());
	}
	
	workers = builder.finish();
}

const kj::Executor& LightWorkerPool::select() {
	size_t idx = base + offset;
	idx %= workers.size();
	
	++offset;
		
	if(offset >= workers.size()) {
		offset = 0;
		base = rand();
	}
	
	return workers[idx] -> getExecutor();
}

}