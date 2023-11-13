#include "async.h"

#include <capnp/dynamic.h>
#include <kj/timer.h>

#include <fsc/data.h>

#include <thread>

#if _WIN32
	#include <winsock2.h>
#else
	#include <sys/socket.h>
#endif

using namespace fscpy;

namespace {
	fsc::LocalDataService& dataService()  {
		return fscpy::PythonContext::libraryThread()->dataService();
	}

	void atExitFunction() {
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
}

PythonWaitScope::~PythonWaitScope() {
	activeScope = nullptr;
}

void PythonWaitScope::turnLoop() {	
	KJ_REQUIRE(activeScope != nullptr, "No WaitScope active");
	KJ_REQUIRE(
		!activeScope->isFiber,
		"Calling the asyncio event loop from a fiber is not safe, because"
		" the fiber needs to yield, which would likely leave the asyncio event"
		" loop in an invalid state. In a fiber, use the fsc.asnc.wait() method"
		" instead, which handles this case properly"
	);
	
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
	KJ_REQUIRE(instance == nullptr, "Can not have two asyncio event ports active in the same thread");
	instance = this;
	
	py::tuple socketPair = py::module_::import("socket").attr("socketpair")();
	readSocket = socketPair[0];
	readSocket.attr("setblocking")(false);
	writeSocket = py::cast<kj::LowLevelAsyncIoProvider::Fd>(socketPair[1].attr("detach")());
	
	// wakeHelper = kj::heap<WakeHelper>(*this);
	
	loopRunner = py::cpp_function([](py::object future) {
		// Either we have no active event loop, or this is a reentrant
		// call to the C++ event loop from a C++ wait. Case 1 there
		// is nothing to do, case 2 the event loop is already turning.
		if(!PythonWaitScope::canWait())
			return;
		
		if(AsyncioEventPort::instance -> waitingForEvents)
			return;
		
		// Try to turn the event loop
		PythonWaitScope::turnLoop();
		
		// Set up a new runner & ready promise
		// Don't do this when called externally since that messes with
		// the externally used ready future.
		AsyncioEventPort::instance -> prepareRunner();
	});
	
	remoteRunner = py::cpp_function([](py::object future) {
		if(AsyncioEventPort::instance == nullptr)
			return;
		
		if(py::bool_(future.attr("cancelled")()))
			return;
		
		AsyncioEventPort::instance -> armRunner();
	});
	
	auto asyncio = py::module_::import("asyncio");
	auto newLoop = asyncio.attr("get_event_loop")();
	
	adjustEventLoop(newLoop);
}

AsyncioEventPort::~AsyncioEventPort() {
	py::gil_scoped_acquire withGil;
		
	instance = nullptr;
	
	#if _WIN32
		closesocket(writeSocket);
	#else
		close(writeSocket);
	#endif
}

void AsyncioEventPort::prepareRunner() {
	readyFuture = eventLoop.attr("create_future")();
	activeRunner = readyFuture.attr("add_done_callback")(loopRunner);
	
	if(listenTask.ptr() != nullptr) {
		listenTask.attr("cancel")();
	}
	listenTask = eventLoop.attr("create_task")(eventLoop.attr("sock_recv")(readSocket, 64));
	activeRemoteRunner = listenTask.attr("add_done_callback")(remoteRunner);
	
	// It can happen that the old "ready" future was fired by unprocessed cross-thread events.
	// If that is the case, we need to manually kick the event loop to make sure this gets handled
	// correctly.
	if(woken || runnable) {
		armRunner();
	}
}

bool AsyncioEventPort::poll() {
	// Let's take an opportunity for a context switch
	{
		py::gil_scoped_release releaseGil;
	}
		
	if(woken) {
		woken = false;
		return true;
	}
	
	return false;
}

bool AsyncioEventPort::wait() {
	/*
	Currently the event port does not get its runnable state adjusted while waiting (because
	the event loop does not set it to false before calling wait() and therefore rising flanks
	are missed too). Therefore, we can not reliably restart the event loop while inside
	EventPort::wait(). For that reason, the waiting logic is instead implemented in
	PythonWaitScope, which has very similar behavior, but suspends in a scope where the port
	correctly receives all enablement notifications.
	*/
	KJ_UNIMPLEMENTED("Waiting from C++ through event port directly is not safe");
	
	/*
	// Jump back into the python event loop until there
	// is nothing left to do.
	instance -> eventLoop.attr("run_until_complete")(instance -> readyFuture);
		
	return instance -> poll();
	*/
}

void AsyncioEventPort::waitForEvents() {
	KJ_REQUIRE(instance != nullptr, "No asyncio event port active");
	// Jump back into the python event loop until there
	// is work for the C++ loop.
	
	auto loop = py::module_::import("asyncio").attr("get_event_loop")();
	adjustEventLoop(mv(loop));
	
	if(py::bool_(instance -> eventLoop.attr("is_closed")()))
		return;
	
	bool restoreTo = instance -> waitingForEvents;
	instance -> waitingForEvents = true;
	KJ_DEFER({ instance -> waitingForEvents = restoreTo; });
	
	instance -> eventLoop.attr("run_until_complete")(instance -> readyFuture);
	instance -> prepareRunner();
}

void AsyncioEventPort::setRunnable(bool newVal) {	
	//KJ_DBG("EvtPort::setRunnable", newVal);
	runnable = newVal;
	
	if(newVal) {
		armRunner();
	}
}

void AsyncioEventPort::wake() const {
	if(!woken.exchange(true)) {
		// wakeHelper -> notify();
		
		// Note: This call uses POSIX or winsock2 API depending on platform
		char bytes[1] = {0};
		send(writeSocket, bytes, 1, 0);
	}
}

void AsyncioEventPort::armRunner() {
	if(!py::bool_(readyFuture.attr("done")())) {
		// Don't arm runner if associated event loop is already closed (no point anyway)
		if(py::bool_(readyFuture.attr("get_loop")().attr("is_closed")())) {
			return;
		}
		readyFuture.attr("set_result")(py::none());
	}
}

void AsyncioEventPort::adjustEventLoop(py::object newLoop) {
	KJ_REQUIRE(instance != nullptr, "No asyncio event port active");
	
	if(instance -> eventLoop.ptr() == newLoop.ptr())
		return;
	
	KJ_REQUIRE(!instance -> waitingForEvents, "Can not switch to a new event loop while already waiting for another");
	
	py::module_::import("nest_asyncio").attr("apply")(newLoop);
	
	instance -> eventLoop = newLoop;
	instance -> prepareRunner();
	instance -> armRunner();
}

// class PythonContext

PythonContext::Instance::Instance() :
	eventPort(),
	thread(library() -> newThread(eventPort)),
	rootScope(thread -> waitScope())
{
}

PythonContext::Instance::~Instance() {
	if(!Py_IsInitialized()) {
		KJ_LOG(FATAL, "Calling thread-local cleanup after python interpreter has shutdown");
	}
}

static kj::MutexGuarded<bool> pythonInitialized(false);

PythonContext::Instance& PythonContext::getInstance() {
	if(!Py_IsInitialized()) {
		KJ_LOG(FATAL, "Calling PythonContext::getInstance without python interpreter");
	}
	
	KJ_REQUIRE(PyGILState_Check(), "Can only check instance while holding GIL");
	
	/* If fusionsc is used in the main thread, its thread local state will not be deleted
	   when Python terminates, instead getting deleted at the end of the executable, after
	   python has been finalized. This can cause the main thread shutdown to hang or crash.
	   
	   To mitigate this issue, we deregister the thread-local data while Python is still
	   active.
	*/
	{
		auto locked = pythonInitialized.lockExclusive();
		
		if(!*locked) {
			*locked = true;
			
			py::module_::import("atexit").attr("register")(
				py::cpp_function([]() {					
					try {
						PythonContext::stop();
					} catch(py::error_already_set& e) {
						KJ_LOG(WARNING, "Exception during cleanup", convertPyError(e));
					}
					
					*pythonInitialized.lockExclusive() = false;
				})
			);
		}
	}
	
	KJ_IF_MAYBE(pInstance, instance) {
		return *pInstance;
	}
	
	return instance.emplace();
}
	
Library PythonContext::library() {
	StartupParameters ctx;
	static kj::MutexGuarded<Library> lib = kj::MutexGuarded<Library>(newLibrary(ctx));
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
			
			try {
				auto result = future.attr("result")();
				parent.fulfiller.fulfill(mv(result));
			} catch(py::error_already_set& e) {
				parent.fulfiller.reject(convertPyError(e));
			}
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
	
	~AsyncioFutureAdapter() {
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
	};
	
	OneOf<Running, Cancelled, py::object, kj::Exception> contents;
	Maybe<Promise<void>> resolveTask;
	
	py::object loop;
	py::list doneCallbacks;
	bool asyncioFutureBlocking = false; 
	
	kj::Canceler canceler;
	
	AsyncioFutureLike(py::object loop, Promise<py::object> promise) :
		contents(py::object()), loop(mv(loop)), doneCallbacks()
	{
		auto forked = promise.fork();
		resolveTask = forked.addBranch().then(
			[this](py::object result) mutable {
				contents = mv(result);
				whenDone();
			},
			[this](kj::Exception e) mutable {
				contents = e;
				whenDone();
			}
		).eagerlyEvaluate(nullptr);
		
		Running r = { mv(forked) };
		contents = mv(r);
	}
	
	AsyncioFutureLike(AsyncioFutureLike& other) :
		AsyncioFutureLike(other.loop, other.asPromise())
	{
	}
	
	virtual ~AsyncioFutureLike() {
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
	
	void addDoneCallback(py::object cb, py::object context) {
		if(context.is_none()) {
			context = py::reinterpret_steal<py::object>(PyContext_CopyCurrent());
		}

		doneCallbacks.append(py::make_tuple(mv(cb), mv(context), this));
		
		if(done())
			whenDone();
	}
	
	uint64_t removeDoneCallback(py::object cb) {
		py::list newCallbacks;
		for(py::handle e : doneCallbacks) {
			auto asTuple = py::reinterpret_borrow<py::tuple>(e);
			
			if(cb.is(asTuple[0]))
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
		
		resolveTask = nullptr;
		canceler.cancel(msgStr);
		contents.init<Cancelled>(mv(msgStr));
		
		whenDone();
		return true;
	}
	
	py::object exception() {
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
		auto exceptionObject = py::reinterpret_borrow<py::object>(eas.value());
		return exceptionObject;
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
		
		AsyncioFutureLike& send(py::object shouldBeNone) {
			if(!consumed) {
				consumed = true;
				parent.asyncioFutureBlocking = true;
				return parent;
			}
			
			KJ_REQUIRE(!parent.asyncioFutureBlocking, "Future was not consumed by asyncio");
			KJ_REQUIRE(parent.done(), "Future was not completed");
			
			auto result = parent.result();
			
			// Throw a StopIteration containing the reflected value given by asyncio
			auto stopIterCls = py::reinterpret_borrow<py::object>(PyExc_StopIteration);
			PyErr_SetObject(PyExc_StopIteration, stopIterCls(result).ptr());
			
			throw py::error_already_set();
		}
		
		AsyncioFutureLike& next() {
			return send(py::none());
		}
		
		void throw_(py::object excType, py::object value, py::object stackTrace) {
			KJ_REQUIRE(consumed, "Protocol error: Exception passed into unconsumed iterator on step 1");
			
			// Test if first object is type or value
			if(PyType_Check(excType.ptr())) {
				if(py::type::of(value).ptr() != excType.ptr()) {
					value = excType(value);
				}
			} else {
				value = excType;
				excType = py::type::of(value);
			}
			
			if(stackTrace.is_none()) {
				stackTrace = value.attr("__traceback__");
			} else {
				PyException_SetTraceback(value.ptr(), stackTrace.ptr());
			}
			
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
		if(py::bool_(loop.attr("is_closed")()))
			return;
		
		auto scheduler = loop.attr("call_soon");
		
		for(py::handle e : doneCallbacks) {
			auto asTuple = py::reinterpret_borrow<py::tuple>(e);
			
			scheduler(asTuple[0], this, py::arg("context") = asTuple[1]);
		}
		
		doneCallbacks = py::list();
	}
};

struct RemotePromise : public AsyncioFutureLike {
	DynamicStructPipeline pipeline;
	
	RemotePromise(py::object loop, Promise<py::object> promise, DynamicStructPipeline pipeline) :
		AsyncioFutureLike(mv(loop), mv(promise)),
		pipeline(mv(pipeline))
	{}
	
	DynamicStructPipeline getPipeline() {
		return pipeline;
	}
};

}

Promise<py::object> adaptAsyncioFuture(py::object future) {
	py::detail::make_caster<AsyncioFutureLike> caster;
	if(caster.load(future, false)) {
		return ((AsyncioFutureLike&) caster).asPromise();
	}
			
	// Python awaitables can be wrapped in tasks
	if(!hasattr(future, "_asyncio_future_blocking")) {
		auto aio = py::module_::import("asyncio");
		auto loop = aio.attr("get_event_loop")();
		future = aio.attr("ensure_future")(mv(future), py::arg("loop") = loop);
		// future = loop.attr("create_task")(.attr("__await__")());
	}
	
	Promise<py::object> adaptedPromise = kj::newAdaptedPromise<py::object, AsyncioFutureAdapter>(mv(future));
	adaptedPromise = getActiveThread().lifetimeScope().wrap(mv(adaptedPromise));
	return adaptedPromise;
}

py::object convertToAsyncioFuture(Promise<py::object> promise) {
	promise = getActiveThread().lifetimeScope().wrap(mv(promise));
	auto loop = py::module_::import("asyncio").attr("get_event_loop")();
	auto result = py::cast(
		new AsyncioFutureLike(loop, mv(promise)),
		py::return_value_policy::take_ownership
	);
	
	// It might be that the asyncio port is currently using a different 
	// event loop. If the asyncio loop is in charge, and the KJ loop
	// transitions to 'ready', the readyness signal will be sent to the
	// wrong event loop, and the KJ loop will never be started by
	// asyncio. To avoid this, we need to adjust now, when we have the
	// correct loop at hand.
	AsyncioEventPort::adjustEventLoop(loop);
	
	return result;
}

py::object convertCallPromise(Promise<py::object> promise, DynamicStructPipeline pipeline) {
	promise = getActiveThread().lifetimeScope().wrap(mv(promise));
	auto loop = py::module_::import("asyncio").attr("get_event_loop")();
	auto result = py::cast(
		new RemotePromise(loop, mv(promise), mv(pipeline)),
		py::return_value_policy::take_ownership
	);
	
	// It might be that the asyncio port is currently using a different 
	// event loop. If the asyncio loop is in charge, and the KJ loop
	// transitions to 'ready', the readyness signal will be sent to the
	// wrong event loop, and the KJ loop will never be started by
	// asyncio. To avoid this, we need to adjust now, when we have the
	// correct loop at hand.
	AsyncioEventPort::adjustEventLoop(loop);
	
	return result;
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
	
	// Constructor
	futureCls.def(py::init<AsyncioFutureLike&>());
	
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
		.def("add_done_callback", &AsyncioFutureLike::addDoneCallback, py::arg("callback"), py::kw_only(), py::arg("context") = py::none())
		.def("remove_done_callback", &AsyncioFutureLike::removeDoneCallback)
		.def("cancel", &AsyncioFutureLike::cancel, py::arg("msg") = py::none())
		.def("exception", &AsyncioFutureLike::exception)
		.def("get_loop", &AsyncioFutureLike::getLoop)
		
		.def("__await__", &AsyncioFutureLike::await, py::keep_alive<0, 1>())
		
		.def_readwrite("_asyncio_future_blocking", &AsyncioFutureLike::asyncioFutureBlocking)
	;
	
	py::class_<AsyncioFutureLike::FutureIterator>(futureCls, "_Iterator")
		.def("__next__", &AsyncioFutureLike::FutureIterator::next, py::return_value_policy::take_ownership)
		.def("send", &AsyncioFutureLike::FutureIterator::send, py::return_value_policy::take_ownership)
		.def("throw", &AsyncioFutureLike::FutureIterator::throw_, py::arg("type"), py::arg("value") = py::none(), py::arg("traceback") = py::none())
		
		.def("__iter__", [](AsyncioFutureLike::FutureIterator& it) -> AsyncioFutureLike::FutureIterator& {
			return it;
		})
	;
	
	py::class_<RemotePromise, AsyncioFutureLike>(asyncModule, "PromiseForResult")
		.def_property_readonly("pipeline", &RemotePromise::getPipeline)
		.def_property_readonly("p", &RemotePromise::getPipeline)
	;
	
	asyncModule.def("startEventLoop", &PythonContext::start, "If the active thread has no active event loop, starts a new one");
	asyncModule.def("stopEventLoop", &PythonContext::stop, "Stops the event loop on this thread if it is active.");
	asyncModule.def("hasEventLoop", &PythonContext::active, "Checks whether this thread has an active event loop");
	asyncModule.def("cycle", &PythonWaitScope::turnLoop, "Turns the C++ event loop until it becomes empty.");
	
	asyncModule.def("canWait", &PythonWaitScope::canWait);
	asyncModule.def("wait", [](Promise<py::object> promise) {
		auto result = PythonWaitScope::wait(mv(promise));
		return result;
	});
		
	auto atexitModule = py::module_::import("atexit");
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}

}