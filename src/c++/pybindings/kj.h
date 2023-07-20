#pragma once

// This file gets included by fscpy.h

namespace fscpy {
	
struct DynamicConstArray {
	virtual size_t size() = 0;
	virtual py::object get(size_t i) = 0;
	virtual ~DynamicConstArray() = 0;
};
	
struct DynamicArray : public DynamicConstArray {
	virtual void set(size_t i, py::object val) = 0;
};

//! Raises an KJ exception in python
void raiseInPython(const kj::Exception& e);

//! Converts a captured python exception to a KJ error
kj::Exception convertPyError(py::error_already_set&);

extern py::exception<kj::Exception> excOverloaded;
extern py::exception<kj::Exception> excDisconnected;
extern py::exception<kj::Exception> excUnimplemented;

}