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
			
			PyErr_SetObject(PyExc_StopIteration, value.ptr());
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