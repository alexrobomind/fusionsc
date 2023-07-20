#pragma once

#include "fscpy.h"

#include <fsc/local.h>

#include <functional>
#include <type_traits>

namespace fscpy {

struct PythonWaitScope {
	PythonWaitScope(kj::WaitScope& ws, bool fiber = false);
	~PythonWaitScope();
	
	template<typename T>
	static T wait(Promise<T>&& promise);
	
	static bool canWait();
	static void turnLoop();
	
private:
	kj::WaitScope& waitScope;
	bool isFiber;
	
	static inline thread_local PythonWaitScope* activeScope = nullptr;
	
	py::object asyncioLoop;
};

struct AsyncioEventPort : public kj::EventPort {
	AsyncioEventPort();
	~AsyncioEventPort();
	
	inline bool wait() override { KJ_UNIMPLEMENTED("Waiting from the C++ side is currently unsupported in asyncio-integrated threads"); }
	
	bool poll() override;
	
	void setRunnable(bool) override;
	void wake() const override;
	
private:
	void cancelRunner();
	
	void scheduleRunner();
	void scheduleRunner() const ;
	
	bool runnable = false;
	
	py::object eventLoop;
	py::object loopRunner;
	
	mutable py::object activeRunner; // Protected by GIL
	mutable bool woken = false;      // Protected by GIL
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