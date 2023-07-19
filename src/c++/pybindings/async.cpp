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
	
	void cycle() {
		fscpy::PythonContext::libraryThread()->waitScope().poll();
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
	 * The coroutine runner of FusionSC currently only accepts promises, runs them via the event loop,
	 * and then feeds back the result object. That means for await contexts on promises, we need the
	 * following procedure:
	 * - The initial next() / send(None) call must return the promise itself to be unwrapped.
	 * - A second send() call will return the result object, which must be "returned" with a StopIteration
	 *   exception.
	 */	 
	struct PyPromiseAwaitContext {
		PyPromise promise;
		bool consumed = false;
		
		PyPromiseAwaitContext(PyPromise& p) :
			promise(p)
		{}
		
		PyPromise send(py::object value) {
			if(!consumed) {
				consumed = true;
				return promise;
			}
			
			// py::print("Coroutine await result", value);
			
			// PySetObject has a lot of weird and partially undocumented behavior, which might change in future versions
			// The safest approach is to actually construct a PyExc_StopIteration object directly
			auto stopIterCls = py::reinterpret_borrow<py::object>(PyExc_StopIteration);
			PyErr_SetObject(PyExc_StopIteration, stopIterCls(mv(value)).ptr());
			
			throw py::error_already_set();
		}
		
		PyPromise throw_(py::object excType, py::object value, py::object stackTrace) {
			if(!consumed) {
				consumed = true;
				return promise;
			}
			
			PyErr_Restore(excType.inc_ref().ptr(), value.inc_ref().ptr(), stackTrace.inc_ref().ptr());
			
			throw py::error_already_set();
		}
		
		PyPromise next() { return send(py::none()); }
	};
	
	/*PyObject* pyPromiseAwait(PyObject* pyObjectPtr) {
		py::handle untypedPromise(pyObjectPtr);
		
		try {
			PyPromise& promise = py::cast<PyPromise&>(untypedPromise);
			return py::cast(new PyPromiseAwaitContext(promise)).inc_ref().get();
		} catch(py::cast_error) {
			PyErr_SetString(PyExc_TypeError, "Internal error: pyPromiseAwait called on an object that is not a PyPromise");
		} catch(std::exception e) {
			PyErr_SetString(PyExc_RuntimeError, "Internal error: Unknown error while interpreting object as PyPromise");
		}
		
		return nullptr;
	}*/
	
	/*void makePyPromiseAwaitable(PyHeapTypeObject *heap_type) {
		// Ensure the PyAsyncMethods struct is initialized
		if(heapType -> tp_as_async == nullptr) {
			heapType -> tp_as_async = PyMem_Malloc(sizeof(PyAsyncMethods));
			memset(heapType -> tp_as_async, 0, sizeof(PyAsyncMethods));
		}
		
		heapType -> tp_as_async -> am_await = &pyPromiseAwait;
	}*/
	
	/** Helper class to cycle through a coroutine
	  * Interprets each yielded value as promise and returns
	  * the unpacked result.
	  */
	struct RunIterator {
		py::object send;
		py::object throw_;
		
		RunIterator(py::object send, py::object throw_) : send(mv(send)), throw_(mv(throw_)) {}
		
		Promise<Own<PyObjectHolder>> doSend(py::object input) {
			py::object argTuple = py::make_tuple(input);
			
			// py::print("ArgTuple for send", argTuple);
			PyObject* sendReturn = PyObject_Call(send.ptr(), argTuple.ptr(), nullptr);
			
			return handleReturn(sendReturn);
		}
		
		Promise<Own<PyObjectHolder>> doThrow(kj::Exception e) {
			// KJ_DBG("Passing error into awaitable", e);
			py::object excType = py::reinterpret_borrow<py::object>(PyExc_RuntimeError);
			
			py::object argTuple = py::make_tuple(mv(excType), py::cast(kj::str(e)), py::none());
			PyObject* sendReturn = PyObject_Call(throw_.ptr(), argTuple.ptr(), nullptr);
			
			return handleReturn(sendReturn);
		}
		
		Promise<Own<PyObjectHolder>> handleReturn(PyObject* sendReturn) {
			// Check for errors
			if(PyErr_Occurred()) {
				KJ_REQUIRE(sendReturn == nullptr, "Internal error: Result non-null despite error indicator");
				
				// This fetches & clears the python error indicator
				py::error_already_set error;
				// KJ_DBG("Handling python error", error.what());
							
				// Check if we have a StopIteration error (which indicates completion)	
				if(PyErr_GivenExceptionMatches(error.type().ptr(), PyExc_StopIteration)) {
					KJ_REQUIRE((bool) error.value(), "Internal error: result null in stop iteration");
					
					// py::print("Received coroutine result", (py::object) error.value().attr("value"));
					return kj::refcounted<PyObjectHolder>((py::object) error.value().attr("value"));
				}
				
				/*py::print("Exception occurred");
				py::print(error.type());
				py::print(error.what());*/
				
				kj::throwFatalException(convertPyError(error));
			}
			
			// We have yielded a value
			py::object nextVal = py::reinterpret_steal<py::object>(sendReturn);
			
			// Make sure it is of correct type (promise)
			KJ_REQUIRE(py::isinstance<PyPromise>(nextVal), "The FSC event loop can only handle its own promise type");
			auto pyPromise = py::cast<PyPromise>(nextVal);
			
			// Wait for yielded promise, then unpack it and forward the result
			// or error to the coroutine.
			return pyPromise
			.then(
				[this](py::object input) mutable {
					py::gil_scoped_acquire withGIL;
					// py::print("Received object", input);
					return this->doSend(mv(input));
				},
				[this](kj::Exception e) mutable {
					py::gil_scoped_acquire withGIL;
					return this->doThrow(mv(e));
				}
			);
		}
		
		~RunIterator() {
			py::gil_scoped_acquire withGIL;
			send = py::object();
			throw_ = py::object();
		}	
	};
	
	PyPromise startFiber(kj::FiberPool& fiberPool, py::object callable) {
		auto func = [callable = mv(callable)](kj::WaitScope& ws) mutable -> PyPromise {			
			// Override default wait scope
			ScopeOverride overrideWS(ws);
			
			py::gil_scoped_acquire withGIL;
			
			// Delete object while in GIL scope
			KJ_DEFER({callable = py::object();});
			
			return callable();
		};
		
		return fiberPool.startFiber(mv(func));
	}
}

namespace fscpy {

// class AsyncioEventPort

AsyncioEventPort::AsyncioEventPort() {
	eventLoop = py::module_::import("asyncio").attr("get_event_loop")();
	loopRunner = py::cpp_function([this]() {
		activeRunner = py::none();
		Promise<void> p = NEVER_DONE;
		PythonWaitScope::poll(p);
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
	}
	
	//! Iterator to satisfy the iterator protocol
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

void raiseInPython(const kj::Exception& e) {
	// TODO: This is extremely primitive and custom exception types would be
	// warranted.
	
	PyErr_SetString(PyExc_RuntimeError, kj::str(e).cStr());
}
	
PyPromise run(PythonAwaitable obj) {
	py::object generator = obj.await();
	
	auto runIt = heapHeld<RunIterator>(generator.attr("send"), generator.attr("throw"));
	
	// Wrap the run to catch exceptions thrown
	// on the first send (and defer the execution)
	auto delayedRun = [runIt]() mutable {
		py::gil_scoped_acquire withGIL;
		return runIt->doSend(py::none());
	};
	
	return kj::evalLater(mv(delayedRun)).attach(runIt.x());
}

void initAsync(py::module_& m) {
	py::module_ asyncModule = m.def_submodule("asnc", "Asynchronous event-loop and promises");
	
	py::type pyGeneric = py::module::import("typing").attr("Generic");
	py::type pyTypeVar = py::module::import("typing").attr("TypeVar");
	
	py::object promiseParam = pyTypeVar("T", py::arg("covariant") = true);
	
	py::class_<PyPromise>(asyncModule, "Promise", py::multiple_inheritance(), py::metaclass(*baseMetaType), R"(
		Asynchronous promise to a future value being computed by the event loop. The computation will eventually
		either resolve or throw an exception. The promise can either be awaited using the wait() method, or using
		the await keyword in asynchronous functions. The "poll" method can be used to check the promise for completion
		(or failure) without blocking, while the "then" function can be used to register a continuation function
		(though we recommend the usage of "await" and coroutines).
	)")
		.def(py::init([](PyPromise& other) { return PyPromise(other); }))
		.def(py::init([](py::object o) { return PyPromise(o); }))
		.def("wait", &PyPromise::wait)
		.def("poll", &PyPromise::poll)
		.def("then", &PyPromise::pyThen)
		
		.def("__await__", [](PyPromise& self) { return new PyPromiseAwaitContext(self); })
		
		// Allows expressions like "Promise[T]"
		.def_static("__class_getitem__", [pyGeneric](py::object key) {
			return pyGeneric.attr("__dict__")["__class_getitem__"].attr("__get__")(py::none(), py::type::of<PyPromise>())(key);
		})
		.def_property_readonly_static("__parameters__", [promiseParam](py::object cls) {
			return py::make_tuple(promiseParam);
		})
	;
	
	py::implicitly_convertible<PythonAwaitable, PyPromise>();
	
	py::class_<PyPromiseAwaitContext>(asyncModule, "_PromiseAwaitCtx")
		.def("__next__", &PyPromiseAwaitContext::next)
		.def("send", &PyPromiseAwaitContext::send)
		.def("throw", &PyPromiseAwaitContext::throw_)
	;
	
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
	
	py::class_<AsyncioFutureLike> futureCls(asyncModule, "Future", "AsyncIO-style future");
	
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
	
	asyncModule.def("startEventLoop", &PythonContext::startEventLoop, "If the active thread has no active event loop, starts a new one");
	asyncModule.def("stopEventLoop", &PythonContext::stopEventLoop, "Stops the event loop on this thread if it is active.");
	asyncModule.def("hasEventLoop", &PythonContext::hasEventLoop, "Checks whether this thread has an active event loop");
	asyncModule.def("cycle", &cycle, "Cycles this thread's event loop a single time");
	
	asyncModule.def("canWait", &PythonWaitScope::canWait);
	
	asyncModule.def("run", &run, "Turns an awaitable (e.g. Promise or Coroutine) into a promise by running it on the active event loop");
	
	auto atexitModule = py::module_::import("atexit");
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}

kj::Exception convertPyError(py::error_already_set& e) {	
	auto formatException = py::module_::import("traceback").attr("format_exception");
	try {
		py::object t = e.type();
		py::object v = e.value();
		py::object tr = e.trace();
		
		py::list formatted;
		if(t && tr)
			formatted = formatException(t, v, tr);
		else
			formatted = formatException(v);
		
		auto pythonException = kj::strTree();
		for(auto s : formatted) {
			pythonException = kj::strTree(mv(pythonException), py::cast<kj::StringPtr>(s), "\n");
		}
		
		// KJ_DBG("Formatted an exception as ", pythonException.flatten());
		
		return kj::Exception(::kj::Exception::Type::FAILED, __FILE__, __LINE__, pythonException.flatten());
	} catch(std::exception e2) {
		py::print("Failed to format exception", e.type(), e.value());
		auto exc = kj::getCaughtExceptionAsKj();
		return KJ_EXCEPTION(FAILED, "An underlying python exception could not be formatted due to the following error", exc);
	}
}
	
}