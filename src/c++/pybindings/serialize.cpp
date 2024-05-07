#include "fscpy.h"
#include "loader.h"
#include "tensor.h"

#include "serialize.h"

#include <capnp/schema.capnp.h>

namespace fscpy {
	py::object loadStructArray(WithMessage<DynamicObject::StructArray::Reader> structArray) {
		auto typeReader = structArray.getSchema().as<capnp::schema::Type>();
		auto structSchema = defaultLoader.capnpLoader.getType(typeReader).asStruct();
		
		auto dataIn = structArray.getData();
		auto shapeIn = structArray.getShape();
		
		kj::Vector<npy_intp> npyShape;
		for(auto e : shapeIn) npyShape.add(e);
		
		PyObject* npyArray = PyArray_SimpleNew(npyShape.size(), npyShape.begin(), NPY_OBJECT);
		if(npyArray == nullptr)
			throw py::error_already_set();
		
		PyObject** npyData = reinterpret_cast<PyObject**>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(npyArray)));
		for(auto i : kj::indices(dataIn)) {
			auto asStruct = dataIn[i].getTarget().as<capnp::DynamicStruct>(structSchema);
			auto asObject = py::cast(DynamicStructReader(shareMessage(structArray), asStruct));
			
			PyObject*& output = *(npyData + i);
			output = asObject.ptr();
			Py_INCREF(output);
		}
		
		return py::reinterpret_steal<py::object>(npyArray);
	}
	
	py::object loadEnumArray(WithMessage<DynamicObject::EnumArray::Reader> structArray) {
		auto typeReader = structArray.getSchema().as<capnp::schema::Type>();
		auto enumSchema = defaultLoader.capnpLoader.getType(typeReader).asEnum();
		
		auto dataIn = structArray.getData();
		auto shapeIn = structArray.getShape();
		
		kj::Vector<npy_intp> npyShape;
		for(auto e : shapeIn) npyShape.add(e);
		
		PyObject* npyArray = PyArray_SimpleNew(npyShape.size(), npyShape.begin(), NPY_OBJECT);
		if(npyArray == nullptr)
			throw py::error_already_set();
		
		PyObject** npyData = reinterpret_cast<PyObject**>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(npyArray)));
		for(auto i : kj::indices(dataIn)) {
			auto asEnum = EnumInterface(enumSchema, dataIn[i]);
			auto asObject = py::cast(asEnum);
			
			PyObject*& output = *(npyData + i);
			output = asObject.ptr();
			Py_INCREF(output);
		}
		
		return py::reinterpret_steal<py::object>(npyArray);
	}
}