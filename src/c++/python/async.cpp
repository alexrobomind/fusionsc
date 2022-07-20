#include "async.h"

#include <capnp/dynamic.h>
#include <kj/timer.h>

#include <fsc/data.h>

using namespace fscpy;

namespace {
	fsc::LocalDataService& dataService()  {
		return fscpy::PyContext::libraryThread()->dataService();
	}

	void atExitFunction() {
		fscpy::PyContext::library()->stopSteward();
	}
	
	void cycle() {
		fscpy::PyContext::libraryThread()->waitScope().poll();
	}
	
	PyPromise readyPromise(py::object o) {
		return fscpy::PyPromise(o);
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
			
			PyErr_Restore(excType.ptr(), value.ptr(), stackTrace.ptr());
			
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
							
				// Check if we have a StopIteration error (which indicates completion)	
				if(PyErr_GivenExceptionMatches(error.type().ptr(), PyExc_StopIteration)) {
					KJ_REQUIRE((bool) error.value(), "Internal error: result null in stop iteration");
					
					return kj::refcounted<PyObjectHolder>((py::object) error.value().attr("value"));
				}
				
				throw error;
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
	
	PyPromise run(py::object obj) {
		// Return promises as is
		try {
			return py::cast<PyPromise>(obj);
		} catch(py::cast_error) {
		}
		
		KJ_REQUIRE(py::hasattr(obj, "__await__"), "Object must be awaitable");
		py::object generator = obj.attr("__await__")();
		
		auto runIt = heapHeld<RunIterator>(generator.attr("send"), generator.attr("throw"));
		
		// Wrap the run to catch exceptions thrown
		// on the first send (and defer the execution)
		auto delayedRun = [runIt]() mutable {
			py::gil_scoped_acquire withGIL;
			return runIt->doSend(py::none());
		};
		
		return kj::evalLater(mv(delayedRun)).attach(runIt.x());
	}
	
	Promise<void> delay(double seconds) {
		uint64_t timeInNS = (uint64_t) seconds * 1e9;
		auto targetPoint = kj::systemPreciseMonotonicClock().now() + timeInNS * kj::NANOSECONDS;
		
		return getActiveThread().timer().atTime(targetPoint);
	}
}

namespace fscpy {

void initAsync(py::module_& m) {	
	py::class_<PyPromise>(m, "Promise", py::multiple_inheritance(), py::metaclass(*baseMetaType)/*, py::custom_type_setup(&makePyPromiseAwaitable)*/)
		.def(py::init([](PyPromise& other) { return PyPromise(other); }))
		.def("wait", &PyPromise::wait)
		.def("poll", &PyPromise::poll)
		.def("then", &PyPromise::pyThen)
		
		.def("__await__", [](PyPromise& self) { return new PyPromiseAwaitContext(self); })
	;
	
	py::class_<PyPromiseAwaitContext>(m, "_PromiseAwaitCtx")
		.def("__next__", &PyPromiseAwaitContext::next)
		.def("send", &PyPromiseAwaitContext::send)
	;
	
	m.def("startEventLoop", &PyContext::startEventLoop);
	m.def("hasEventLoop", &PyContext::hasEventLoop);
	m.def("dataService", &dataService);
	m.def("cycle", &cycle);
	m.def("readyPromise", &readyPromise);
	m.def("run", &run);
	m.def("delay", &delay);
	
	auto atexitModule = py::module_::import("atexit");
	atexitModule.attr("register")(py::cpp_function(&atExitFunction));
}
	
}