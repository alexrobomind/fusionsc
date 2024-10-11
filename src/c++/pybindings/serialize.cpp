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
		size_t shapeProd = 1;
		for(auto e : shapeIn) { npyShape.add(e); shapeProd *= e; }
		KJ_REQUIRE(dataIn.size() == shapeProd);
		
		PyObject* npyArray = PyArray_SimpleNew(npyShape.size(), npyShape.begin(), NPY_OBJECT);
		if(npyArray == nullptr)
			throw py::error_already_set();
		
		PyObject** npyData = reinterpret_cast<PyObject**>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(npyArray)));
		for(auto i : kj::indices(dataIn)) {
			auto asStruct = dataIn[i].getTarget().as<capnp::DynamicStruct>(structSchema);
			auto asObject = py::cast(DynamicStructReader(shareMessage(structArray), asStruct));
			
			npyData[i] = asObject.release().ptr();
		}
		
		return py::reinterpret_steal<py::object>(npyArray);
	}
	
	py::object loadEnumArray(WithMessage<DynamicObject::EnumArray::Reader> structArray) {
		auto typeReader = structArray.getSchema().as<capnp::schema::Type>();
		auto enumSchema = defaultLoader.capnpLoader.getType(typeReader).asEnum();
		
		capnp::List<uint16_t>::Reader dataIn = structArray.getData();
		auto shapeIn = structArray.getShape();
		
		auto enumerants = enumSchema.getEnumerants();
		auto knownEnumerants = [&]() {
			auto builder = kj::heapArrayBuilder<py::object>(enumerants.size());
			for(auto& e : enumerants) builder.add(py::cast(EnumInterface(e)));
			return builder.finish();
		}();
		
		kj::HashMap<uint64_t, py::object> unknownEnumerants;
		
		auto getEnumerant = [&](uint16_t raw) -> py::object {
			if(raw < knownEnumerants.size())
				return knownEnumerants[raw];
			
			KJ_IF_MAYBE(pEntry, unknownEnumerants.find(raw)) {
				return *pEntry;
			}
			
			py::object newVal = py::cast(EnumInterface(enumSchema, raw));
			unknownEnumerants.insert(raw, newVal);
			return newVal;
		};
		
		kj::Vector<npy_intp> npyShape;
		size_t shapeProd = 1;
		for(auto e : shapeIn) { npyShape.add(e); shapeProd *= e; }
		KJ_REQUIRE(dataIn.size() == shapeProd);
		
		PyObject* npyArray = PyArray_SimpleNew(npyShape.size(), npyShape.begin(), NPY_OBJECT);
		if(npyArray == nullptr)
			throw py::error_already_set();
		
		PyObject** npyData = reinterpret_cast<PyObject**>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(npyArray)));
		for(auto i : kj::indices(dataIn)) {
			npyData[i] = getEnumerant(dataIn[i]).release().ptr();
		}
		
		return py::reinterpret_steal<py::object>(npyArray);
	}
}