#pragma once

#include "fscpy.h"

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
	
	static inline LibraryThread libraryThread() {
		ensureLT();
		return _libraryThread->addRef();
	}
	
	static inline kj::WaitScope& waitScope() {
		ensureLT();
		return _libraryThread -> waitScope();
	}
	
private:
	static inline kj::MutexGuarded<Library> _library;
	static inline thread_local LibraryThread _libraryThread;
	
	static inline void ensureLT() {
		if(_libraryThread.get() == nullptr) {
			_libraryThread = library() -> newThread();
		}
	}
};

struct PyPromise {
	inline PyPromise(Promise<py::object> input) :
		holder(input.fork())
	{}
	
	inline PyPromise(PyPromise& other) :
		holder(other.holder.addBranch().fork())
	{}
	
	inline Promise<py::object> get() {
		return holder.addBranch();
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
		py::gil_scoped_release release_gil;
		return holder.addBranch().wait(PyContext::waitScope());
	}
	
	inline bool poll() {
		py::gil_scoped_release release_gil;
		return holder.addBranch().poll(PyContext::waitScope());
	}
	
private:
	kj::ForkedPromise<py::object> holder;
};

}