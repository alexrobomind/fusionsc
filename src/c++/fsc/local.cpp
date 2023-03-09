#include <list>

#include "local.h"
#include "data.h"

namespace fsc {
	
// === class LibraryHandle ===

void LibraryHandle::stopSteward() const {
	_daemonRunner->disconnect();
	storeSteward.stop();
}

std::unique_ptr<Botan::HashFunction> LibraryHandle::defaultHash() const {
	auto result = Botan::HashFunction::create("Blake2b");
	KJ_REQUIRE(result != nullptr, "Requested hash function not available");
	return result;
}
	
// === class ThreadHandle ===

namespace {

struct NullErrorHandler : public kj::TaskSet::ErrorHandler {
	static inline NullErrorHandler instance;
	
	void taskFailed(kj::Exception&& e) {}
};

}

struct ThreadHandle::Ref {		
	Ref(const kj::MutexGuarded<RefData>* refData);
	~Ref();
		
	kj::ListLink<Ref> link;
	const kj::MutexGuarded<RefData>* refData;
};

struct ThreadHandle::RefData {
	kj::List<Ref, &Ref::link> refs;
	Own<CrossThreadPromiseFulfiller<void>> whenNoRefs;
	const ThreadHandle* owner;
};
		
ThreadHandle::Ref::Ref(const kj::MutexGuarded<RefData>* refData) :
	refData(refData)
{
	refData -> lockExclusive() -> refs.add(*this);
}

ThreadHandle::Ref::~Ref() {
	auto locked = refData -> lockExclusive();
	locked -> refs.remove(*this);
	
	auto& pFulfiller = locked -> whenNoRefs;
	if(pFulfiller.get() != nullptr) {
		pFulfiller->fulfill();
		pFulfiller = nullptr;
	}
}

ThreadHandle::ThreadHandle(Library l) :
	_ioContext(kj::setupAsyncIo()),
	_library(l -> addRef()),
	_executor(kj::getCurrentThreadExecutor()),
	_dataService(kj::heap<LocalDataService>(l)),
	_filesystem(kj::newDiskFilesystem()),
	refData(new kj::MutexGuarded<RefData>()),
	detachedTasks(NullErrorHandler::instance)
{	
	KJ_REQUIRE(current == nullptr, "Can only have one active ThreadHandle / LibraryThread per thread");
	current = this;
}

ThreadHandle::~ThreadHandle() {
	KJ_REQUIRE(current == this, "Destroying LibraryThread in wrong thread") {}
	
	// If the library is in shutdown mode, we have to (conservatively) assume 
	// that all other threads  have unexpectedly died and this is the last
	// remaining active thread. We can not BANK on it, but we can not wait
	// for any other promises on the event loop to resolve.
	if(!_library->inShutdownMode()) {
		while(true) {
			Promise<void> noMoreRefs = READY_NOW;
			
			{
				auto locked = refData -> lockExclusive();
				
				if(locked -> refs.size() == 0)
					break;
				
				auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
				noMoreRefs = mv(paf.promise);
				
				locked -> whenNoRefs = mv(paf.fulfiller);
			}
			
			noMoreRefs.wait(waitScope());
		}
		
		delete refData;
	} else {
		bool canDeleteRefdata = false;
		{
			auto locked = refData -> lockExclusive();
		
			if(locked -> refs.size() == 0)
				canDeleteRefdata = true;
		}
		
		if(canDeleteRefdata)
			delete refData;
	}
			
	current = nullptr;
}

Own<ThreadHandle> ThreadHandle::addRef() {
	return Own<ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}

Own<const ThreadHandle> ThreadHandle::addRef() const {
	return Own<const ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}

void ThreadHandle::detach(Promise<void> p) {
	detachedTasks.add(mv(p));
}

Promise<void> ThreadHandle::drain() {
	return detachedTasks.onEmpty();
}

Promise<void> ThreadHandle::uncancelable(Promise<void> p) {
	auto forked = p.fork();
	detach(p.addBranch());
	return p.addBranch();
}

}