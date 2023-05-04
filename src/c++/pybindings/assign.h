#pragma once

#include <capnp/dynamic.h>
#include <capnp/list.h>

#include "fscpy.h"

namespace fscpy {

struct BuilderSlot {
	mutable capnp::Type type;
	
	BuilderSlot(capnp::Type type) : type(type) {}
	
	virtual void set(capnp::DynamicValue::Reader other) const = 0;
	virtual void adopt(capnp::Orphan<capnp::DynamicValue>&& orphan) const = 0;
	virtual capnp::DynamicValue::Builder get() const = 0;
	virtual capnp::DynamicValue::Builder init() const = 0;
	virtual capnp::DynamicValue::Builder init(unsigned int size) const = 0;
	
	virtual ~BuilderSlot() {};
};

void assign(const BuilderSlot& dst, py::object object);

void assign(capnp::DynamicStruct::Builder dsb, kj::StringPtr fieldName, py::object value);
void assign(capnp::DynamicList::Builder dlb, uint32_t idx, py::object value);

}