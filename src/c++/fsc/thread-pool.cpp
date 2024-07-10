#include "local.h"

#include <thread>

namespace fsc { namespace internal {
	
LightWorkerThread::LightWorkerThread(LibraryHandle& hdl) :
	thread([this, &hdl]() mutable { run(hdl); })
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

void LightWorkerThread::run(LibraryHandle& libHandle) {
	WorkerContext ctx(kj::attachRef(libHandle));
	
	// Create cross thread promise
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	onDestroy = mv(paf.fulfiller);
	
	// Signal constructor we are ready
	*(executor.lockExclusive()) = kj::getCurrentThreadExecutor().addRef();
	
	// Wait until destruction
	paf.promise.wait(ctx.waitScope());
}

const kj::Executor& LightWorkerThread::getExecutor() const {
	auto locked = executor.lockShared();
	
	KJ_IF_MAYBE(pExec, *locked) {
		return **pExec;
	}
	
	KJ_UNREACHABLE;
}

LightWorkerPool::LightWorkerPool(LibraryHandle& hdl, size_t nWorkers) {
	if(nWorkers == 0)
		nWorkers = std::thread::hardware_concurrency();
	
	auto builder = kj::heapArrayBuilder<Own<LightWorkerThread>>(nWorkers);
	
	for(auto i : kj::range(0, nWorkers)) {
		builder.add(kj::heap<LightWorkerThread>(hdl));
	}
	
	workers = builder.finish();
}

const kj::Executor& LightWorkerPool::select() const {
	size_t idx = base.load() + (offset++);
	idx %= workers.size();
		
	if(offset.load() >= workers.size()) {
		offset = 0;
		base = rand();
	}
	
	return workers[idx] -> getExecutor();
}

}}