#include "assign.h"
#include "tensor.h"
#include "async.h"
#include "structio.h"

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
	DynamicValue::Builder init() const override {
		auto t = field.getType();
		
		if(t.isStruct()) {
			return builder.init(field);
		}
		
		builder.clear(field);
		return builder.get(field);
	}
	DynamicValue::Builder init(unsigned int size) const override { return builder.init(field, size); }
	void clear() const override {
		builder.clear(field);
	}
};

struct ListItemSlot : public BuilderSlot {
	mutable DynamicList::Builder list;
	mutable uint32_t index;
	
	ListItemSlot(DynamicList::Builder list, uint32_t index) :
		BuilderSlot(list.getSchema().getElementType()),
		list(list), index(index)
	{}
	
	void set(DynamicValue::Reader newVal) const override {
		if(list.getSchema().getElementType().isStruct() && newVal.getType() == DynamicValue::STRUCT) {
			// Check if the list can hold all data
			capnp::AnyStruct::Builder dst = list[index].as<DynamicStruct>();
			capnp::AnyStruct::Reader src = newVal.as<DynamicStruct>();
			
			size_t srcDataSize = src.getDataSection().size();
			size_t dstDataSize = dst.getDataSection().size();
			size_t srcPointerSize = src.getPointerSection().size();
			size_t dstPointerSize = dst.getPointerSection().size();
			
			auto logWarning = [](kj::StringPtr msg) {
				auto warnings = py::module_::import("warnings");
				warnings.attr("warn")(msg.cStr());
			};
			
			if(dstDataSize < srcDataSize) {
				bool isSet = false;
				
				for(const byte b : src.getDataSection().slice(dstDataSize, srcDataSize))
					if(b != 0) isSet = true;
				
				if(isSet) {
					logWarning(kj::str(
						"You are assigning a struct into a list whose struct size is smaller "
						"than the source list. The extra data will be truncated and will be "
						"missing. Target data size: ", dstDataSize, " bytes, source "
						"data size: ", srcDataSize, " bytes."
					));
				}
			}
			
			if(dstPointerSize < srcPointerSize) {
				bool isSet = false;
				
				auto ps = src.getPointerSection();
				for(auto i : kj::range(dstPointerSize, srcPointerSize)) {
					auto p = ps[i];
					if(!p.isNull())
						isSet = true;
				}					
				
				if(isSet) {
					logWarning(kj::str(
						"You are assigning a struct into a list whose struct size is smaller "
						"than the source list. Extra pointers will be truncated and will be "
						"missing. Target pointer size: ", dstPointerSize, " words, Src "
						"pointer size: ", srcPointerSize, " words."
					));
				}
			}
			
		}
		list.set(index, newVal);
	}
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
	void clear() const override { init(); }
	
};

struct MessageSlot : public BuilderSlot {
	capnp::MessageBuilder& message;
	mutable capnp::StructSchema schema;
	
	MessageSlot(capnp::MessageBuilder& nMessage, capnp::StructSchema nSchema) :
		BuilderSlot(nSchema),
		message(nMessage), schema(nSchema)
	{}
	
	void set(DynamicValue::Reader newVal) const override {
		KJ_REQUIRE(newVal.getType() == DynamicValue::STRUCT, "Can only assign struct values to messages");
		auto structVal = newVal.as<capnp::DynamicStruct>();
		
		KJ_REQUIRE(structVal.getSchema() == schema);
		message.setRoot(structVal);
	}
	void adopt(capnp::Orphan<DynamicValue>&& orphan) const override {
		KJ_REQUIRE(orphan.getType() == DynamicValue::STRUCT, "Can only assign struct values to messages");
		auto structOrphan = orphan.releaseAs<capnp::DynamicStruct>();
		
		KJ_REQUIRE(structOrphan.get().getSchema() == schema);
		message.adoptRoot(mv(structOrphan));
	}
	DynamicValue::Builder get() const override {
		return message.getRoot<capnp::DynamicStruct>(schema);
	}
	
	DynamicValue::Builder init() const override {
		return message.initRoot<capnp::DynamicStruct>(schema);
	}
	
	DynamicValue::Builder init(unsigned int size) const override {
		KJ_FAIL_REQUIRE("Messages may not be initialized with list content");
	}
	void clear() const override {
		init();
	}
};

struct CapSlot : public BuilderSlot {
	mutable DynamicCapability::Client cap;
	
	CapSlot(capnp::InterfaceSchema schema) :
		BuilderSlot(schema)
	{
		capnp::Capability::Client raw = nullptr;
		cap = raw.castAs<capnp::DynamicCapability>(schema);
	}
	
	~CapSlot() noexcept {}
	
	void set(DynamicValue::Reader newVal) const override {
		KJ_REQUIRE(newVal.getType() == DynamicValue::CAPABILITY, "Can only assign capabilities to capability slots");
		auto newCap = newVal.as<capnp::DynamicCapability>();
		KJ_REQUIRE(newCap.getSchema().extends(cap.getSchema()));
		cap = mv(newCap);
	}
	void adopt(capnp::Orphan<DynamicValue>&& orphan) const override {
		KJ_REQUIRE(orphan.getType() == DynamicValue::CAPABILITY, "Can only assign capabilities to capability slots");
		auto newCap = orphan.releaseAs<capnp::DynamicCapability>();
		KJ_REQUIRE(newCap.get().getSchema() == cap.getSchema());
		cap = newCap.get();
	}
	DynamicValue::Builder get() const override {
		return cap;
	}
	DynamicValue::Builder init() const override {
		capnp::Capability::Client raw = nullptr;
		cap = raw.castAs<capnp::DynamicCapability>(cap.getSchema());
		return cap;
	}
	
	DynamicValue::Builder init(unsigned int size) const override {
		KJ_FAIL_REQUIRE("Capabilities may not be initialized with list content");
	}
	
	void clear() const override { init(); }
};

}

void assign(const BuilderSlot& dst, py::object object) {
	auto assignmentFailureLog = kj::strTree();
	
	/*// Attempt 0: Check if assignable
	if(py::isinstance<Assignable>(object)) {
		py::cast<Assignable&>(object).assign(dst);
		return;
	}*/
	// Check if object is wrapper
	if(py::hasattr(object, "_fusionsc_wraps")) {
		assign(dst, object.attr("_fusionsc_wraps"));
		return;
	}
	
	// Attempt 1: Check if target is orphan that can be adopted
	pybind11::detail::make_caster<DynamicOrphan> orphanCaster;
	if(orphanCaster.load(object, false)) {
		DynamicOrphan& orphan = (DynamicOrphan&) orphanCaster;
		dst.adopt(orphan.release());
		return;
	}
	
	// Attempt 2: Try to parse structs / lists as YAML
	pybind11::detail::make_caster<kj::StringPtr> strCaster;
	if(py::isinstance<py::str>(object) && (dst.type.isList() || dst.type.isStruct())) {
		KJ_IF_MAYBE(pException, kj::runCatchingExceptions([&]() {
			// Create destination visitor
			Own<::fsc::structio::Visitor> v;
			auto t = dst.type;			
			if(t.isList()) {
				auto initializer = [&](size_t s) {
					return dst.init(s).as<capnp::DynamicList>();
				};
				v = ::fsc::structio::createVisitor(t.asList(), initializer);
			} else {
				v = ::fsc::structio::createVisitor(dst.init().as<capnp::DynamicStruct>());
			}
			
			// Load buffer
			Py_ssize_t size;
			const char* utf8 = PyUnicode_AsUTF8AndSize(object.ptr(), &size);
			if(utf8 == nullptr)
				throw py::error_already_set();
			kj::ArrayPtr<const kj::byte> bufferPtr((const kj::byte*) utf8, size);
						
			// Perform load
			::fsc::structio::load(bufferPtr, *v, ::fsc::structio::Dialect::YAML);
			return;
		})) {
			auto& error = *pException;
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Error while trying to assign from YAML: ", error, "\n");
		} else {
			return;
		}
	}
	
	// Attempt 3: Check if target can be converted into a reader directly
	pybind11::detail::make_caster<DynamicValueReader> dynValCaster;
	if(dynValCaster.load(object, false)) {
		try {
			auto& val = (DynamicValueReader&) dynValCaster;
			
			// Assigning from void in our case means clearing the field / resetting to default
			if(val.getType() == capnp::DynamicValue::VOID) {
				// dst.init();
				dst.clear();
			} else {
				dst.set(val);
			}
			
			return;
		} catch(kj::Exception e) {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Error while trying to assign from primitive: ", e, "\n");
		}
	}
	
	if(PyErr_Occurred()) {
		throw py::error_already_set();
	}
	
	// Attempt 4: Try to assign from a dict
	if(py::dict::check_(object) && dst.type.isStruct()) {
		auto asDict = py::reinterpret_borrow<py::dict>(object);
		auto dstAsStruct = dst.init().as<DynamicStruct>();
		
		for(auto item : asDict) {
			pybind11::detail::make_caster<kj::StringPtr> keyCaster;
			if(!keyCaster.load(item.first, true)) {
				assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from dict because a key could not be converted to string\n");
				goto dict_assignment_failed;
			}
			
			kj::StringPtr fieldName = static_cast<kj::StringPtr>(keyCaster);
			
			auto maybeField = dst.type.asStruct().findFieldByName(fieldName);
			KJ_IF_MAYBE(pField, maybeField) {
				try {
					assign(FieldSlot(dstAsStruct, *pField), py::reinterpret_borrow<py::object>(item.second));
				} catch(std::exception e) {
					assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning from dict because assignment of field failed\nField name: ", fieldName, "\nError: ", e.what(), "\n");
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
	
	// Attempt 7: Capabilities can also be assigned from promises
	if(dst.type.isInterface()) {
		using PromiseType = Promise<DynamicCapabilityClient>;
		py::detail::make_caster<PromiseType> caster;
		if(caster.load(object, false)) {
			auto targetSchema = dst.type.asInterface();
			
			capnp::Capability::Client promisedCap = static_cast<PromiseType&>(caster)
			.then([targetSchema](DynamicCapabilityClient obj) -> capnp::Capability::Client {
				auto newSchema = obj.getSchema();
				KJ_REQUIRE(newSchema.extends(targetSchema), "Returned capability does not extend the required interface");
				
				return obj;
			});
			
			dst.set(promisedCap.castAs<capnp::DynamicCapability>(targetSchema));
		} else {
			assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning capability from future because object is no future\n");
		}
	} else {
		assignmentFailureLog = strTree(mv(assignmentFailureLog), "Skipped assigning capability from future because slot type is not a capability interface\n");
	}
	
	throw std::invalid_argument(kj::str("Could not find a way to assign object of type ", py::cast<kj::StringPtr>(py::str(py::type::of(object))), ".\n", assignmentFailureLog.flatten()).cStr());
}


void assign(capnp::DynamicList::Builder list, uint32_t index, py::object value) {
	assign(ListItemSlot(list.as<DynamicList>(), index), mv(value));
}

void assign(capnp::DynamicStruct::Builder builder, kj::StringPtr fieldName, py::object value) {
	assign(FieldSlot(builder, builder.getSchema().getFieldByName(fieldName)), mv(value));
}

void assign(capnp::MessageBuilder& builder, capnp::StructSchema schema, py::object value) {
	assign(MessageSlot(builder, mv(schema)), value);
}

}