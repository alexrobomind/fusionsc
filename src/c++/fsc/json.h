#pragma once

#include "common.h"

#include <capnp/dynamic.h>
#include <kj/io.h>

namespace fsc {

void loadJson(capnp::DynamicStruct::Builder, kj::BufferedInputStream&);
void loadJson(capnp::ListSchema, kj::Function<capnp::DynamicList::Builder(size_t)>, kj::BufferedInputStream&);

void loadCbor(capnp::DynamicStruct::Builder, kj::BufferedInputStream&);
void loadCbor(capnp::ListSchema, kj::Function<capnp::DynamicList::Builder(size_t)>, kj::BufferedInputStream&);

capnp::DynamicValue::Reader loadJsonPrimitive(capnp::Type, kj::BufferedInputStream&);
capnp::DynamicValue::Reader loadCborPrimitive(capnp::Type, kj::BufferedInputStream&);

void writeCbor(capnp::DynamicValue::Reader, kj::BufferedOutputStream&);
void writeJson(capnp::DynamicValue::Reader, kj::BufferedOutputStream&, bool strict = true);

}