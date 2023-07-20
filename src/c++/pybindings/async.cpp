#include "async.h"

#include <capnp/dynamic.h>
#include <kj/timer.h>

#include <fsc/data.h>

using namespace fscpy;

namespace {
	fsc::LocalDataService& dataService()  {
		return fscpy::PythonContext::libraryThread()->dataService();
	}

	void atExitFunction() {
		fscpy::PythonContext::library()->setShutdownMode();
		fscpy::PythonContext::library()->stopSteward();
	}
	
	Promise<py::object> startFiber(kj::FiberPool& fiberPool, py::object callable) {
		auto func = [callable = mv(callable)](kj::WaitScope& ws) mutable -> py::object {
			// Since we run the event loop inside the GIL now, we do not need to
			// acquire the GIL again for waiting.
			// py::gil_scoped_acquire withGIL;		
			
			// Override default wait scope
			PythonWaitScope overrideWS(ws, true);
			
			// Delete object while in GIL scope
			KJ_DEFER({callable = py::object();});
			
			return callable();
		};
		
		return fiberPool.startFiber(mv(func));
	}
}

namespace fscpy {

// class PythonWaitScope

PythonWaitScope::PythonWaitScope(kj::WaitScope& ws, bool fiber) : waitScope(ws), isFiber(fiber) {
	KJ_REQUIRE(
		activeScope == nullptr,
		"Trying to allocate a new PyWaitScope while another one is active."
		" This is most likely an error internal to fusionsc, because it means that a C++-side code waited"
		" on an event loop promise without releasing the active python scope"
	);
	activeScope = this;
	asyncioLoop = py::module_::import("asyncio").attr("get_event_loop")();
}

PythonWaitScope::~PythonWaitScope() {
	activeScope = nullptr;
}

void PythonWaitScope::turnLoop() {	
	KJ_REQUIRE(
		activeScope != nullptr,
		"Trying to turn the event loop outside an active wait scope."
		" This means that either no event loop was started on this thread,"
		" or that the event loop is currently already turning and this function"
		" was called from the inside of a coroutine driven by the C++ loop."
	);
	KJ_REQUIRE(!activeScope->isFiber, "Can not turn the event loop inside a fiber");
	
	auto restoreTo = activeScope;
	activeScope = nullptr;
	KJ_DEFER({activeScope = restoreTo;});
	
	Promise<void> p = NEVER_DONE;
	p.poll(restoreTo->waitScope);
}

bool PythonWaitScope::canWait() {
	return activeScope != nullptr;
}

// class AsyncioEventPort

AsyncioEventPort::AsyncioEventPort() {
	eventLoop = py::module_::import("asyncio").attr("get_event_loop")();
	loopRunner = py::cpp_function([this]() {
		activeRunner = py::none();
		PythonWaitScope::turnLoop();
	});
	activeRunner = py::none();
}

AsyncioEventPort::~AsyncioEventPort() {
	py::gil_scoped_acquire withGil;
	
	// Running the loopRunner after the event port is destroyed
	// is a really really bad idea.
	cancelRunner();
}

bool AsyncioEventPort::poll() {
	// Let's take an opportunity for a context switch
	{
		py::gil_scoped_release releaseGil;
	}
	
	py::gil_scoped_acquire withGil;
	
	if(woken) {
		woken = false;
		return true;
	}
	
	return false;
}

void AsyncioEventPort::setRunnable(bool newVal) {
	py::gil_scoped_acquire withGil;
	
	runnable = newVal;
	
	if(newVal) {
		scheduleRunner();
	}
}

void AsyncioEventPort::wake() const {
	py::gil_scoped_acquire withGil;
	
	if(!woken) {
		woken = true; 
		scheduleRunner();
	}
}

void AsyncioEventPort::cancelRunner() {
	if(!activeRunner.is_none()) {
		activeRunner.attr("cancel")();
		activeRunner = py::none();
	}
}

void AsyncioEventPort::scheduleRunner() {
	if(activeRunner.is_none()) {
		activeRunner = eventLoop.attr("call_soon")(loopRunner);
	}
}

void AsyncioEventPort::scheduleRunner() const {
	if(activeRunner.is_none()) {
		activeRunner = eventLoop.attr("call_soon_threadsafe")(loopRunner);
	}
}

// class PythonContext

PythonContext::Instance::Instance() :
	thread(library() -> newThread()),
	rootScope(thread -> waitScope())
{}

PythonContext::Instance& PythonContext::getInstance() {
	KJ_IF_MAYBE(pInstance, instance) {
		return *pInstance;
	}
	
	return instance.emplace();
}
	
Library PythonContext::library() {
	static kj::MutexGuarded<Library> lib = kj::MutexGuarded<Library>(newLibrary(true));
	return (**lib.lockExclusive()).addRef();
}

void PythonContext::start() {
	getInstance();
}

void PythonContext::stop() {
	instance = nullptr;
}

bool PythonContext::active() {
	return instance != nullptr;
}

LibraryThread& PythonContext::libraryThread() {
	return getInstance().thread;
}

// class AsyncioFutureAdapter

namespace {

struct AsyncioFutureAdapter {
	struct FulfillerCallback : public kj::Refcounted {
		bool valid = true;
		AsyncioFutureAdapter& parent;
		
		void call(py::object future) {
			if(!valid)
				return;
			
			parent.fulfiller.rejectIfThrows([this, future]() mutable {
				parent.fulfiller.fulfill(future.attr("result")());
			});
		}
		
		FulfillerCallback(AsyncioFutureAdapter& parent) :
			parent(parent)
		{}
	};
	
	AsyncioFutureAdapter(kj::PromiseFulfiller<py::object>& fulfiller, py::object future):
		fulfiller(fulfiller),
		future(future)
	{
		cppCallback = kj::refcounted<FulfillerCallback>(*this);
		pythonCallback = py::cpp_function([target = kj::addRef(*cppCallback)](py::object future) mutable {
			target -> call(future);
		});
		
		future.attr("add_done_callback")(pythonCallback);
	}
	
	~AsyncioFutureAdapter(){
		future.attr("remove_done_callback")(pythonCallback);
		cppCallback -> valid = false;
	}
	
private:
	PromiseFulfiller<py::object>& fulfiller;
	
	Own<FulfillerCallback> cppCallback;
	py::object pythonCallback;
	py::object future;
};

struct AsyncioFutureLike {
	struct Cancelled {
		kj::String msg;
		
		Cancelled(kj::String msg) :
			msg(mv(msg))
		{}
	};
	
	struct Running {
		ForkedPromise<py::object> directPath;
		Promise<void> resolveTask;
	};
	
	OneOf<Running, Cancelled, py::object, kj::Exception> contents;
	
	py::object loop;
	py::list doneCallbacks;
	bool asyncioFutureBlocking = false; 
	
	kj::Canceler canceler;
	
	AsyncioFutureLike(Promise<py::object> promise) :
		contents(py::object()), doneCallbacks()
	{
		auto forked = promise.fork();
		Promise<void> resolveTask = forked.addBranch().then(
			[this](py::object result) mutable {
				contents = mv(result);
			},
			[this](kj::Exception e) mutable {
				contents = e;
				whenDone();
			}
		);
		
		Running r = { mv(forked), mv(resolveTask) };
		contents = mv(r);
	}
	
	AsyncioFutureLike(AsyncioFutureLike& other) :
		AsyncioFutureLike(other.asPromise())
	{}
	
	~AsyncioFutureLike() {
		canceler.release();
	}
		
	py::object result() {
		if(contents.is<Running>()) {
			py::object excType = py::module_::import("asyncio").attr("InvalidStateError");
			PyErr_SetString(excType.ptr(), "Attempting to access result of incomplete promise");
			throw py::error_already_set();
		} else if(contents.is<py::object>()) {
			return contents.get<py::object>();
		} else if(contents.is<kj::Exception>()) {
			raiseInPython(contents.get<kj::Exception>());
			throw py::error_already_set();
		} else if(contents.is<Cancelled>()) {
			py::object excType = py::module_::import("asyncio").attr("CancelledError");
			PyErr_SetString(excType.ptr(), contents.get<Cancelled>().msg.cStr());
			throw py::error_already_set();
		}
		
		KJ_UNREACHABLE;
	}
	
	void setResult(py::object result) {
		KJ_FAIL_REQUIRE("Setting results directly on promise-derived futures is prohibited");
	}
	
	void setException(py::object exception) {
		KJ_FAIL_REQUIRE("Setting results directly on promise-derived futures is prohibited");
	}
	
	bool done() {
		return !contents.is<Running>();
	}
	
	bool cancelled() {
		return contents.is<Cancelled>();
	}
	
	void addDoneCallback(py::object cb) {
		doneCallbacks.append(mv(cb));
		
		if(done())
			whenDone();
	}
	
	uint64_t removeDoneCallback(py::object cb) {
		py::list newCallbacks;
		for(py::handle e : doneCallbacks) {
			if(e == cb)
				continue;
			newCallbacks.append(e);
		}
		
		uint64_t removed = doneCallbacks.size() - newCallbacks.size();
		doneCallbacks = mv(newCallbacks);
		return removed;
	}
	
	bool cancel(py::object msg) {
		if(done())
			return false;
		
		kj::String msgStr = kj::str("Cancelled");
		
		if(!msg.is_none()) {
			msgStr = kj::heapString(py::cast<kj::StringPtr>(msg));
		}
		
		canceler.cancel(msgStr);
		contents.init<Cancelled>(mv(msgStr));
		
		whenDone();
		return true;
	}
	
	py::handle exception() {
		if(contents.is<Running>()) {
			py::object excType = py::module_::import("asyncio").attr("InvalidStateError");
			PyErr_SetString(excType.ptr(), "Attempting to access result of incomplete promise");
			throw py::error_already_set();
		} else if(contents.is<py::object>()) {
			return py::none();
		} else if(contents.is<kj::Exception>()) {
			raiseInPython(contents.get<kj::Exception>());
		} else if(contents.is<Cancelled>()) {
			py::object excType = py::module_::import("asyncio").attr("CancelledError");
			PyErr_SetString(excType.ptr(), contents.get<Cancelled>().msg.cStr());
			throw py::error_already_set();
		}
		
		py::error_already_set eas;
		return eas.value();
	}
	
	py::object getLoop() {
		return loop;
	}
	
	Promise<py::object> asPromise() {
		if(contents.is<Running>()) {
			return canceler.wrap(contents.get<Running>().directPath.addBranch());
		} else if(contents.is<py::object>()) {
			return contents.get<py::object>();
		} else if(contents.is<kj::Exception>()) {
			return cp(contents.get<kj::Exception>());
		} else if(contents.is<Cancelled>()) {
			return kj::evalNow([this]() -> py::object {
				KJ_FAIL_REQUIRE(contents.get<Cancelled>().msg);
			});
		}
		
		KJ_UNREACHABLE;
	}
	
	/**
	 * Generator to be returned by 'await' statements on PyPromises.
	 *
	 * A little semantics: When calling 'await' in a coroutine, Python obtains a generator
	 * from the awaitable. The generator is then called once (via next()) and the returned
	 * object is passed to the coroutine runner, which sends back (via the outer coroutine-generated
	 * generator the object to be returned in place). The unwrap-and-send procedure continues
	 * as long as the "send" method returns and ends when a "StopIteration" is raised, at which
	 * point the coroutine will take the value given there as return value and continue the coroutine.
	 *
	 * Coroutine runners generally take promise-style objects and then send the result when they have
	 * resolved. This means for the promise itself the iterator must behave as follows:
	 * - The initial next() / send(None) call must return the promise itself to be waited for.
	 * - A second send() call will return the result object, which must be "returned" (reflected into
	 *   the coroutine with a StopIteration exception.
	 */	 
	struct FutureIterator {
		FutureIterator(AsyncioFutureLike& parent) :
			parent(parent)
		{}
		
		AsyncioFutureLike& send(py::object value) {
			if(!consumed) {
				consumed = true;
				parent.asyncioFutureBlocking = true;
				return parent;
			}
			
			KJ_REQUIRE(!parent.asyncioFutureBlocking, "Future was not consumed by asyncio");
			KJ_REQUIRE(parent.done(), "Future was not completed");
			
			// Throw a StopIteration containing the reflected value given by asyncio
			auto stopIterCls = py::reinterpret_borrow<py::object>(PyExc_StopIteration);
			PyErr_SetObject(PyExc_StopIteration, stopIterCls(mv(value)).ptr());
			
			throw py::error_already_set();
		}
		
		AsyncioFutureLike& next() {
			return send(py::none());
		}
		
		void throw_(py::object excType, py::object value, py::object stackTrace) {
			KJ_REQUIRE(consumed, "Protocol error: Exception passed into unconsumed iterator on step 1");
			
			PyErr_Restore(excType.inc_ref().ptr(), value.inc_ref().ptr(), stackTrace.inc_ref().ptr());
			throw py::error_already_set();
		}
	
	private:
		AsyncioFutureLike& parent;
		bool consumed = false;
	};
	
	FutureIterator* await() {
		return new FutureIterator(*this);
	}

private:
	void whenDone() {		
		auto scheduler = loop.attr("call_soon");
		
		for(py::handle e : doneCallbacks)
			scheduler(e);
		
		doneCallbacks = py::list();
	}
};

}

Promise<py::object> adaptAsyncioFuture(py::object future) {
	py::detail::make_caster<AsyncioFutureLike> caster;
	if(caster.load(future, false)) {
		return ((AsyncioFutureLike&) caster).asPromise();
	}
	
	return kj::newAdaptedPromise<py::object, AsyncioFutureAdapter>(mv(future));
}

py::object convertToAsyncioFuture(Promise<py::object> promise) {
	return py::cast(new AsyncioFutureLike(mv(promise)));
}

py::type futureType() {
	return py::type::of<AsyncioFutureLike>();
}

void initAsync(py::module_& m) {
	py::module_ asyncModule = m.def_submodule("asnc", "Asynchronous event-loop and promises");
	
	py::type pyGeneric = py::module::import("typing").attr("Generic");
	py::type pyTypeVar = py::module::import("typing").attr("TypeVar");
	
	py::object promiseParam = pyTypeVar("T", py::arg("covariant") = true);
	
	py::class_<kj::FiberPool>(asyncModule, "FiberPool", R"(
		Despite python's extensive support for coroutines, it can sometimes be neccessary to use the blocking Promise.wait(...)
		function inside a coroutine (e.g. if using an external library, which commonly is not directly compatible with event-
		loops). Since this would block the very event loop that processes the promise, this is not possible.
		
		Fiber pools resolve this issue by allowing to start a callable as a pseudo-thread (referred to as a "fiber"), which
		maintains its own stack, but is otherwise contained in the active thread and follows the same cooperative scheduling.
		Calling "wait" on any promise inside the fiber will suspend the fiber and continue event loop scheduling until the
		promise resolved, at which point the fiber will continue execution. Like with coroutines, the active thread will not
		perform other tasks while fiber is active (which eliminates the need for locking between fibers).
	)")
		.def(py::init<unsigned int>())
		.def("startFiber", &startFiber, py::keep_alive<0, 1>())
	;
	
	py::class_<AsyncioFutureLike> futureCls(
		asyncModule, "Future",
		py::multiple_inheritance(), py::metaclass(*baseMetaType),
		"AsyncIO-style future"
	);
	
	// Define template specialization mechanism
	// Allows expressions like "Future[T]"
	futureCls
		.def_static("__class_getitem__", [pyGeneric](py::object key) {
			return pyGeneric.attr("__dict__")["__class_getitem__"].attr("__get__")(py::none(), py::type::of<AsyncioFutureLike>())(key);
		})
		.def_property_readonly_static("__parameters__", [promiseParam](py::object cls) {
			return py::make_tuple(promiseParam);
		})
	;
	
	futureCls
		.def("result", &AsyncioFutureLike::result)
		.def("set_result", &AsyncioFutureLike::setResult)
		.def("set_exception", &AsyncioFutureLike::setException)
		.def("done", &AsyncioFutureLike::done)
		.def("cancelled", &AsyncioFutureLike::cancelled)
		.def("add_done_callback", &AsyncioFutureLike::addDoneCallback)
		.def("remove_done_callback", &AsyncioFutureLike::removeDoneCallback)
		.def("cancel", &AsyncioFutureLike::cancel)
		.def("exception", &AsyncioFutureLike::exception)
		.def("get_loop", &AsyncioFutureLike::getLoop)
		
		.def("__await__", &AsyncioFutureLike::await, py::keep_alive<0, 1>())
		
		.def_readwrite("_asyncio_future_blocking", &AsyncioFutureLike::asyncioFutureBlocking)
	;
	
	py::class_<AsyncioFutureLike::FutureIterator>(futureCls, "_Iterator")
		.def("__next__", &AsyncioFutureLike::FutureIterator::next)
		.def("send", &AsyncioFutureLike::FutureIterator::send)
		.def("throw", &AsyncioFutureLike::FutureIterator::throw_)
	;
	
	asyncModule.def("startEventLoop", &PythonContext::start, "If the active thread has no active event loop, starts a new one");
	asyncModule.def("stopEventLoop", &PythonContext::stop, "Stops the event loop on this thread if it is active.");
	asyncModule.def("hasEventLoop", &PythonContext::active, "Checks whether this thread has an active event loop");
	asyncModule.def("cycle", &PythonWaitScope::turnLoop, "Turns the C++ event loop until it becomes empty.");
	
	asyncModule.def("canWait", &PythonWaitScope::canWait);
		
	auto atexitModule = py::module_::import("atexit");
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}

}