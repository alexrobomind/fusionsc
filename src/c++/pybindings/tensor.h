#pragma once

#include "fscpy.h"
#include "capnp.h"

#include <capnp/dynamic.h>

// The Numpy API is imported in capnp.cpp

#ifndef FSCPY_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL fscpy_ARRAY_API

#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>

namespace fscpy {
	
bool isTensor(capnp::Type type);
	
PyArray_Descr* numpyWireType(capnp::Type type);
py::buffer getAsBufferViaNumpy(py::object input, capnp::Type type, int minDims, int maxDims);

//! Non-primitive types reference the underlying message, therefore these need their parent to be
// kept alive.
bool needsBackReference(capnp::DynamicValue::Type t);

Tuple<size_t, kj::StringPtr> pyFormat(capnp::Type type);

void setTensor(capnp::DynamicStruct::Builder dsb, py::buffer buffer);

py::buffer_info getTensor(DynamicStructInterface<capnp::DynamicStruct::Reader> self);
py::buffer_info getTensor(DynamicStructInterface<capnp::DynamicStruct::Builder> self);

}