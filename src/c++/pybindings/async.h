#pragma once

#include "fscpy.h"

#include <fsc/local.h>

#include <functional>
#include <type_traits>
#include <atomic>

namespace fscpy {

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
};

struct AsyncioEventPort : public kj::EventPort {
	AsyncioEventPort();
	~AsyncioEventPort();
	
	inline bool wait() override;
	
	bool poll() override;
	
	void setRunnable(bool) override;
	void wake() const override;
	
private:
	void prepareRunner();
	
	void armRunner();
	void armRunner() const ;
	
	bool runnable = false;
	
	py::object eventLoop;
	py::object loopRunner;
	py::object readyFuture;
	
	py::object activeRunner; // Protected by GIL
	
	mutable std::atomic<bool> woken = false;
};

Promise<py::object> adaptAsyncioFuture(py::object future);
py::object convertToAsyncioFuture(Promise<py::object> promise);
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
		
		AsyncioEventPort eventPort;
		LibraryThread thread;
		PythonWaitScope rootScope;
	};
	
	static Instance& getInstance();
	
	static inline kj::MutexGuarded<Library> _library = kj::MutexGuarded<Library>();
	static thread_local inline Maybe<Instance> instance = nullptr;
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