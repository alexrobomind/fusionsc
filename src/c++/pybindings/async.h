#pragma once

#include "fscpy.h"

#include <fsc/local.h>

#include <functional>
#include <type_traits>
#include <atomic>

namespace fscpy {

class AsyncioEventPort;

//! Manages the active wait scope to be used by the asyncio event loop.
struct PythonWaitScope {
	PythonWaitScope(kj::WaitScope& ws, bool fiber = false);
	~PythonWaitScope();
	
	template<typename T>
	static T wait(Promise<T>&& promise);
	
	template<typename P>
	static bool poll(P&& promise);
	
	static bool canWait();
	static void turnLoop();
	
private:
	kj::WaitScope& waitScope;
	bool isFiber;
	
	static inline thread_local PythonWaitScope* activeScope = nullptr;
	
	friend class AsyncioEventPort;
};

struct AsyncioEventPort : public kj::EventPort {
	AsyncioEventPort();
	~AsyncioEventPort();
	
	bool wait() override;
	
	bool poll() override;
	
	void setRunnable(bool) override;
	void wake() const override;
	
	static void waitForEvents();
	static void adjustEventLoop(py::object newLoop);
	
private:
	void prepareRunner();
	
	void armRunner();
	
	bool runnable = false;
	
	py::object eventLoop;
	py::object loopRunner;
	py::object readyFuture;
	
	py::object activeRunner; // Protected by GIL
	
	mutable std::atomic<bool> woken = false;
	
	struct WakeHelper;
	Own<WakeHelper> wakeHelper;
	
	inline static thread_local AsyncioEventPort* instance = nullptr;
	
	friend class PythonWaitScope;
	bool waitingForEvents = false;
};

Promise<py::object> adaptAsyncioFuture(py::object future);
py::object convertToAsyncioFuture(Promise<py::object> promise);
py::object convertCallPromise(Promise<py::object> promise, DynamicStructPipeline pipeline);
py::type futureType();

struct PythonContext {
	static Library library();
	
	static void start();
	static void stop();
	static bool active();
	
	static LibraryThread& libraryThread();
	
private:
	struct Instance {
		Instance();
		~Instance();
		
		AsyncioEventPort eventPort;
		LibraryThread thread;
		PythonWaitScope rootScope;
	};
	
	static Instance& getInstance();
	
	static inline kj::MutexGuarded<Library> _library = kj::MutexGuarded<Library>();
	static thread_local inline Maybe<Instance> instance = nullptr;
	
	static inline py::object atExitCallback;
};

}

namespace pybind11 { namespace detail {

template<>
struct type_caster<kj::Promise<py::object>>;

template<typename T>
struct type_caster<kj::Promise<T>>;

template<>
struct type_caster<kj::Promise<void>>;

}}

#include "async-inl.h"