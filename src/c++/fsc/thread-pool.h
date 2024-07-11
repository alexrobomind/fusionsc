#pragma once

namespace fsc {
	class LibraryHandle;
}

namespace fsc { namespace internal {
	
struct LightWorkerThread {
	LightWorkerThread(LibraryHandle&);
	~LightWorkerThread();
	const kj::Executor& getExecutor() const;
	
private:
	void run(LibraryHandle&);
	
	kj::MutexGuarded<Maybe<Own<const kj::Executor>>> executor;
	Own<kj::CrossThreadPromiseFulfiller<void>> onDestroy;
	
	kj::Thread thread;
};

struct LightWorkerPool {
	LightWorkerPool(LibraryHandle&, size_t numWorkers = 0);
	
	const kj::Executor& select() const;
	
	kj::Array<Own<LightWorkerThread>> workers;

private:
	mutable std::atomic<size_t> offset = 0;
	mutable std::atomic<size_t> base = 0;
};

}}