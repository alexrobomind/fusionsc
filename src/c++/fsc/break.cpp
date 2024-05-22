#include "break.h"
#include "local.h"

#ifdef _WIN32
	#include <windows.h>
	#include <kj/windows-sanity.h>
#else
	#include <kj/async-unix.h>
#endif

#include <kj/async-queue.h>

namespace fsc {

namespace {

#ifdef _WIN32

//! Handles win32 console events
BOOL WINAPI win32ConsoleHandler(_In_ DWORD eventType);

struct BreakHandlerImpl {
	const kj::Executor& executor = kj::getCurrentThreadExecutor();
	
	mutable kj::WaiterQueue<int> queue;
	
	BreakHandlerImpl() {
		KJ_WIN32(SetConsoleCtrlHandler(&win32ConsoleHandler, TRUE));
	}
	
	~BreakHandlerImpl() {
		(void) SetConsoleCtrlHandler(&win32ConsoleHandler, FALSE);
	}
	
	void signal() const {
		executor.executeSync([this]{
			// executeSync synchronizes the threads
			// the lock is held by whever called signal()
			// the queue is only accessed in the main thread
			
			while(!queue.empty()) {
				queue.fulfill(0);
			}
		});
	}
	
	Promise<void> wait() const {
		KJ_REQUIRE(&kj::getCurrentThreadExecutor() == &executor, "This function may only be called in the main thread");
		return queue.wait().ignoreResult();
	}
	
	static void prepare() {}
};

static kj::MutexGuarded<Maybe<BreakHandlerImpl>> handlerImpl = kj::MutexGuarded<Maybe<BreakHandlerImpl>>(nullptr);

BOOL WINAPI win32ConsoleHandler(_In_ DWORD eventType) {
	auto locked = handlerImpl.lockShared();
	
	switch(eventType) {
		case CTRL_C_EVENT:
			KJ_IF_MAYBE(pImpl, *locked) {
				pImpl -> signal();
				return TRUE;
			}
			
		default:
			break;
	}
	
	return FALSE;
}

#else

struct BreakHandlerImpl {
	mutable kj::WaiterQueue<int> queue;
	
	kj::UnixEventPort& unixEventPort;
	const kj::Executor& executor = kj::getCurrentThreadExecutor();
	
	Promise<void> listener;
	
	BreakHandlerImpl() :
		unixEventPort(getActiveThread().ioContext().unixEventPort),
		listener(listen().eagerlyEvaluate(nullptr))
	{}
	
	~BreakHandlerImpl() {
	}
	
	Promise<void> wait() const {
		KJ_REQUIRE(&kj::getCurrentThreadExecutor() == &executor, "This function may only be called in the main thread");
		return queue.wait().ignoreResult();
	}
	
	Promise<void> listen() {
		Promise<siginfo_t> p = NEVER_DONE;
		
		p = p.exclusiveJoin(unixEventPort.onSignal(SIGINT));
		p = p.exclusiveJoin(unixEventPort.onSignal(SIGTERM));
		
		return p.then([this](siginfo_t sigInfo) {
			while(!queue.empty()) {
				queue.fulfill(0);
			}
			
			return listen();
		});
	}
	
	static void prepare() {
		kj::UnixEventPort::captureSignal(SIGTERM);
		kj::UnixEventPort::captureSignal(SIGINT);
	}
};

static kj::MutexGuarded<Maybe<BreakHandlerImpl>> handlerImpl = kj::MutexGuarded<Maybe<BreakHandlerImpl>>(nullptr);
	
#endif

}

BreakHandler::BreakHandler() {
	KJ_REQUIRE(!hasActiveThread(), "Break handler must be created before library startup");
	BreakHandlerImpl::prepare();
}

BreakHandler::~BreakHandler() {
	auto locked = handlerImpl.lockExclusive();
	*locked = nullptr;
}

Promise<void> BreakHandler::onBreak() {
	{
		auto locked = handlerImpl.lockShared();
		KJ_IF_MAYBE(pImpl, *locked) {
			return pImpl -> wait();
		}
	}
	
	{
		auto locked = handlerImpl.lockExclusive();
		KJ_IF_MAYBE(pImpl, *locked) {
			return pImpl -> wait();
		} else {
			auto& impl = locked -> emplace();
			return impl.wait();
		}
	}
}

}