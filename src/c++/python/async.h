#pragma once

#include "fscpy.h"

#include <fsc/local.h>

#include <functional>

namespace fscpy {

struct PyContext {
	static inline Library library() {
		{
			auto locked = _library.lockShared();
			
			if(locked->get() != nullptr)
				return locked->get()->addRef();
		}
		
		auto locked = _library.lockExclusive();
		*locked = newLibrary();
		
		return locked->get()->addRef();
	}
	
	static inline LibraryThread& libraryThread() {
		startEventLoop();
		return _libraryThread;
	}
	
	static inline kj::WaitScope& waitScope() {
		startEventLoop();
		return _libraryThread -> waitScope();
	}
	
	static inline void startEventLoop() {
		if(_libraryThread.get() == nullptr) {
			_libraryThread = library() -> newThread();
		}
	}
	
	static inline void stopEventLoop() {
		_libraryThread = nullptr;
	}
	
	static inline bool hasEventLoop() {
		return _libraryThread.get() != nullptr;
	}
	
	static inline void clearLibrary() {
		auto locked = _library.lockExclusive();
		*locked = nullptr;
	}
	
private:
	static inline kj::MutexGuarded<Library> _library = kj::MutexGuarded<Library>();
	static inline thread_local LibraryThread _libraryThread;
};

struct PyObjectHolder : public kj::Refcounted {
	py::object content;
	
	PyObjectHolder(const PyObjectHolder& other) = delete;
	inline PyObjectHolder(py::object&& newContent) : content(mv(newContent)) {}
	
	Own<PyObjectHolder> addRef() { return kj::addRef(*this); }
	
	inline ~PyObjectHolder() {
		py::gil_scoped_acquire withGIL;
		content = py::object();
	}
};

kj::Exception convertPyError(py::error_already_set& e);

struct PyPromise {
	inline PyPromise(Promise<Own<PyObjectHolder>> input) :
		holder(input.fork())
	{}
	
	inline PyPromise(py::object obj) :
		holder(nullptr)
	{
		auto holder = kj::refcounted<PyObjectHolder>(mv(obj));
		Promise<Own<PyObjectHolder>> holderPromise = mv(holder);
		
		this -> holder = holderPromise.fork();
	}
	
	inline PyPromise(PyPromise& other) :
		holder(other.holder.addBranch().fork())
	{}
	
	inline PyPromise(PyPromise&& other) :
		holder(mv(other.holder))
	{}
	
	template<typename T>
	PyPromise(Promise<T> input) :
		holder(
			input
			.then([](T x) {
				py::gil_scoped_acquire withGIL;
				return kj::refcounted<PyObjectHolder>(py::cast(x));
			})
			.fork()
		)
	{
		static_assert(!kj::isSameType<T, py::object>(), "Promise<py::object> is unsafe");
	}
	
	PyPromise(Promise<void> input) :
		holder(
			input
			.then([]() {
				py::gil_scoped_acquire withGIL;
				return kj::refcounted<PyObjectHolder>(py::none());
			})
			.fork()
		)
	{}
	
	// Direct conversion to / from py::object promises is unsafe
	// The underlying object might get deleted between events,
	// which would happen outside the GIL.
	
	/*inline Promise<py::object> get() {
		return holder.addBranch().then([](Own<PyObjectHolder> holder) {
			py::gil_scoped_acquire withGIL;
			
			py::object increasedRefcount = holder->content;
			return increasedRefcount; // Either move-returned or RVO'd, so this doesnt change refcount and therefore doesnt need GIL
		});
	
		inline operator Promise<py::object>() {
			return get();
		}
	}*/
	
	template<typename F, typename... Args>
	auto then(F&& func, Args&&... args) {
		return holder.addBranch()
		.then(
			[func = fwd<F>(func)](Own<PyObjectHolder> holder) mutable {
				py::gil_scoped_acquire withGIL;
				
				try {
					return func(holder -> content);
				} catch(py::error_already_set& e) {
					kj::throwFatalException(convertPyError(e));
				}
			},
			fwd<Args>(args)...
		);
	}
	
	template<typename F>
	auto catch_(F&& func) {
		return holder.addBranch().catch_(fwd<F>(func));
	}
	
	template<typename T>
	Promise<T> as() {
		static_assert(!kj::isSameType<T, py::object>(), "Promise<py::object> is unsafe");
		
		return then([](py::object pyObj) { return py::cast<T>(pyObj); });
	}
	
	Promise<void> ignoreResult() {
		return holder.addBranch().ignoreResult();
	}
	
	template<typename T>
	operator Promise<T>() {
		return as<T>();
	}
	
	inline py::object wait() {
		Own<PyObjectHolder> pyObjectHolder;
		
		{
			py::gil_scoped_release release_gil;
			pyObjectHolder = holder.addBranch().wait(PyContext::waitScope());
		}
		
		return pyObjectHolder -> content;
	}
	
	inline bool poll() {
		py::gil_scoped_release release_gil;
		return holder.addBranch().poll(PyContext::waitScope());
	}
	
	inline PyPromise pyThen(py::function f) {
		return then([f = mv(f)](py::object o) {
			py::object result = f(o);
			return kj::refcounted<PyObjectHolder>(mv(result));
		});
	}
	
private:
	kj::ForkedPromise<Own<PyObjectHolder>> holder;
};

struct PythonAwaitable {
	py::object object;
	
	inline py::iterator await() { return object.attr("__await__")(); }
	inline operator PyPromise();
};

PyPromise run(PythonAwaitable obj);

PythonAwaitable::operator PyPromise() {	return run(*this); }

}

namespace pybind11 { namespace detail {

template<typename T>
struct type_caster<kj::Promise<T>> {
	using ValueConv = make_caster<T>;
	PYBIND11_TYPE_CASTER(kj::Promise<T>, const_name("fsc.native.asnc.Promise[") + ValueConv::name + const_name("]"));
	
	bool load(handle src, bool convert) {
		type_caster<fscpy::PyPromise> baseCaster;
		if(!baseCaster.load(src, convert))
			return false;
				
		value = static_cast<fscpy::PyPromise&>(baseCaster).as<T>();
		return true;
	}
		
	static handle cast(kj::Promise<T> src, return_value_policy policy, handle parent) {
		return type_caster<fscpy::PyPromise>::cast(fscpy::PyPromise(mv(src)), policy, parent);
	}	
};

template<>
struct type_caster<kj::Promise<void>> {
	PYBIND11_TYPE_CASTER(kj::Promise<void>, const_name("fsc.native.asnc.Promise[NoneType]"));
	
	bool load(handle src, bool convert) {
		type_caster<fscpy::PyPromise> baseCaster;
		if(!baseCaster.load(src, convert))
			return false;
				
		value = static_cast<fscpy::PyPromise&>(baseCaster).ignoreResult();
		return true;
	}
		
	static handle cast(kj::Promise<void> src, return_value_policy policy, handle parent) {
		return type_caster<fscpy::PyPromise>::cast(fscpy::PyPromise(mv(src)), policy, parent);
	}	
};

template<>
struct type_caster<fscpy::PythonAwaitable> {
	PYBIND11_TYPE_CASTER(fscpy::PythonAwaitable, const_name("Awaitable"));
	
	bool load(handle src, bool convert) {
		py::type srcType = py::type::of(src);
		PyTypeObject* pyType = reinterpret_cast<PyTypeObject*>(srcType.ptr());
		
		PyAsyncMethods* asyncMethods = pyType -> tp_as_async;
		
		// Check if python async struct is present
		if(asyncMethods == nullptr)
			return false;
		
		if(asyncMethods->am_await == nullptr)
			return false;
		
		value.object = py::reinterpret_borrow<py::object>(src);
		return true;
	}
	
	static handle cast(fscpy::PythonAwaitable src, return_value_policy policy, handle parent) {
		return src.object.inc_ref();
	}
};

}}