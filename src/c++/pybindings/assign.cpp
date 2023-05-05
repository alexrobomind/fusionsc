#include "assign.h"
#include "tensor.h"

#include <fsc/yaml.h>

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;

namespace py = pybind11;

namespace fscpy {

namespace {

void setTensor(DynamicValue::Builder dvb, py::buffer buffer) {
	//TODO: Derive tensor type from buffer value?
	::fscpy::setTensor(dvb.as<DynamicStruct>(), buffer);
}

struct FieldSlot : public BuilderSlot {
	mutable DynamicStruct::Builder builder;
	mutable capnp::StructSchema::Field field;
	
	FieldSlot(DynamicStruct::Builder builder, capnp::StructSchema::Field field) :
		BuilderSlot(field.getType()),
		builder(builder), field(field)
	{}
	
	void set(DynamicValue::Reader newVal) const override { builder.set(field, newVal); }
	void adopt(capnp::Orphan<DynamicValue>&& orphan) const override { builder.adopt(field, mv(orphan)); }
	DynamicValue::Builder get() const override { return builder.get(field); }
	DynamicValue::Builder init() const override { return builder.init(field); }
	DynamicValue::Builder init(unsigned int size) const override { return builder.init(field, size); }
};

struct ListItemSlot : public BuilderSlot {
	mutable DynamicList::Builder list;
	mutable uint32_t index;
	
	ListItemSlot(DynamicList::Builder list, uint32_t index) :
		BuilderSlot(list.getSchema().getElementType()),
		list(list), index(index)
	{}
	
	void set(DynamicValue::Reader newVal) const override { list.set(index, newVal); }
	void adopt(capnp::Orphan<DynamicValue>&& orphan) const override { list.adopt(index, mv(orphan)); }
	DynamicValue::Builder get() const override { return list[index]; }
	
	DynamicValue::Builder init() const override { 
		// Unfortunately, DynamicList::Builder has no init(idx) method
		// Therefore, we need to hack one up ourselves
		// This is only relevant for struct cases, so let's first check for that
		if(!list.getSchema().getElementType().isStruct())
			return get();
		
		// Construct a default value and assign it
		capnp::MallocMessageBuilder tmp;
		auto tmpRoot = tmp.initRoot<capnp::DynamicStruct>(list.getSchema().getElementType().asStruct());
		list.set(index, tmpRoot.asReader());
		
		return get();
	}
	
	DynamicValue::Builder init(unsigned int size) const override { return list.init(index, size); }
	
};

}

void assign(const BuilderSlot& dst, py::object object) {
	auto assignmentFailureLog = kj::strTree();
	
	// Attempt 1: Check if target is orphan that can be adopted
	pybind11::detail::make_caster<capnp::Orphan<DynamicValue>> orphanCaster;
	if(orphanCaster.load(object, false)) {
		capnp::Orphan<DynamicValue>& orphanRef = (capnp::Orphan<DynamicValue>&) orphanCaster;
		dst.adopt(mv(orphanRef));
		return;
	}
	
	// Attempt 2: Try to parse structs / lists as YAML
	pybind11::detail::make_caster<kj::StringPtr> strCaster;
	if(strCaster.load(object, false) && (dst.type.isList() || dst.type.isStruct())) {
		KJ_IF_MAYBE(pException, kj::runCatchingExceptions([&]() {
			auto node = YAML::Load(((kj::StringPtr) strCaster).cStr());
			
			if(dst.type.isList()) {
				auto asList = dst.init(node.size()).as<capnp::DynamicList>();
				load(asList, node);
				return;
			} else if(dst.type.isStruct()) {
				auto asStruct = dst.init().as<capnp::DynamicStruct>();
				load(asStruct, node);
				return;
			}
		})) {
			auto& error = *pException;
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Error while trying to assign from YAML: ", error, "\n");
		} else {
			return;
		}
	}
	
	// Attempt 3: Check if target can be converted into a reader directly
	pybind11::detail::make_caster<DynamicValue::Reader> dynValCaster;
	if(dynValCaster.load(object, false)) {
		try {
			dst.set((DynamicValue::Reader&) dynValCaster);
			return;
		} catch(kj::Exception e) {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Error while trying to assign from primitive: ", e, "\n");
		}
	}
	
	// Attempt 4: Try to assign from a dict
	if(py::dict::check_(object) && dst.type.isStruct()) {
		auto asDict = py::reinterpret_borrow<py::dict>(object);
		auto dstAsStruct = dst.get().as<DynamicStruct>();
		
		for(auto item : asDict) {
			pybind11::detail::make_caster<kj::StringPtr> keyCaster;
			if(!keyCaster.load(item.first, false)) {
				assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from dict because a key could not be converted to string\n");
				goto dict_assignment_failed;
			}
			
			kj::StringPtr fieldName = static_cast<kj::StringPtr>(keyCaster);
			
			auto maybeField = dst.type.asStruct().findFieldByName(fieldName);
			KJ_IF_MAYBE(pField, maybeField) {
				try {
					assign(FieldSlot(dstAsStruct, *pField), py::reinterpret_borrow<py::object>(item.second));
				} catch(std::exception e) {
					assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from dict because assignment of field failed\n", fieldName, e.what());
					goto dict_assignment_failed;
				}
			} else {
				assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from dict because a key did not have a corresponding field\n", fieldName);
				goto dict_assignment_failed;
			}
		}
		
		return;
	} else {
		if(!dst.type.isStruct()) {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from dict because slot type is not struct\n");
		}
	}
	
	dict_assignment_failed:
	
	// Attempt 5: Try to assign from a sequence
	if(py::sequence::check_(object) && dst.type.isList()) {
		auto asSequence = py::reinterpret_borrow<py::sequence>(object);
		
		DynamicList::Builder listDst = dst.init(asSequence.size()).as<DynamicList>();
		for(unsigned int i = 0; i < listDst.size(); ++i) {
			assign(ListItemSlot(listDst, i), asSequence[i]);
		}
		
		return;
	} else {
		if(!dst.type.isList()) {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from sequence because slot type is not list\n");
		}
	}
	
	// Attempt 6: If we are a tensor, try to convert via a numpy array
	if(isTensor(dst.type)) {
		auto scalarType = dst.type.asStruct().getFieldByName("data").getType().asList().getElementType();
		
		py::buffer targetBuffer;
		try {
			targetBuffer = getAsBufferViaNumpy(object, scalarType, 0, 100);
		} catch(py::error_already_set& e) {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Could not obtain buffer from numpy due to following reason: ", e.what(), "\n");
			goto tensor_conversion_failed;
		}
		
		// From now on, we don't wanna catch exceptions, as this should
		// always work
		setTensor(dst.get(), targetBuffer);
		return;
	} else {
		assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from array because slot type is not a tensor\n");
	}
	
	// This label exists in case we add more conversion routines later
	tensor_conversion_failed:
	
	throw std::invalid_argument(kj::str("Could not find a way to assign object of type ", py::cast<kj::StringPtr>(py::str(py::type::of(object))), ".\n", assignmentFailureLog.flatten()).cStr());
}


void assign(capnp::DynamicList::Builder list, uint32_t index, py::object value) {
	assign(ListItemSlot(list.as<DynamicList>(), index), mv(value));
}

void assign(capnp::DynamicStruct::Builder builder, kj::StringPtr fieldName, py::object value) {
	assign(FieldSlot(builder, builder.getSchema().getFieldByName(fieldName)), mv(value));
}

}