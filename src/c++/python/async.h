#pragma once

#include "fscpy.h"

#include <fsc/local.h>

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
	PyObjectHolder(py::object&& newContent) : content(mv(newContent)) {}
	
	Own<PyObjectHolder> addRef() { return kj::addRef(*this); }
};

struct PyPromise {
	inline PyPromise(Promise<py::object> input) :
		holder(input.then([](py::object input) { return kj::refcounted<PyObjectHolder>(mv(input)); }).fork())
	{}
	
	inline PyPromise(PyPromise& other) :
		holder(other.holder.addBranch().fork())
	{}
	
	inline Promise<py::object> get() {
		return holder.addBranch().then([](Own<PyObjectHolder> holder) {
			py::gil_scoped_acquire withGIL;
			
			py::object increasedRefcount = holder->content;
			return increasedRefcount; // Either move-returned or RVO'd, so this doesnt change refcount and therefore doesnt need GIL
		});
	}
	
	inline operator Promise<py::object>() {
		return get();
	}
	
	template<typename T>
	Promise<T> as() {
		return get().then([](py::object pyObj) { return py::cast<T>(pyObj); });
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
	
private:
	kj::ForkedPromise<Own<PyObjectHolder>> holder;
};

}