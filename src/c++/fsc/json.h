#pragma once

#include "common.h"

#include <capnp/dynamic.h>
#include <kj/io.h>

namespace fsc {

void loadJson(capnp::DynamicStruct::Builder, kj::InputStream&);
void loadCbor(capnp::DynamicList::Builder, kj::InputStream&);

void loadCbor(capnp::DynamicStruct::Builder, kj::InputStream&);
void loadCbor(capnp::DynamicList::Builder, kj::InputStream&);

capnp::DynamicValue::Reader loadJsonPrimitive(capnp::Type, kj::InputStream&);
capnp::DynamicValue::Reader loadCborPrimitive(capnp::Type, kj::InputStream&);

void writeCbor(capnp::DynamicStruct::Reader, kj::OutputStream&);
void writeCbor(capnp::DynamicList::Reader, kj::OutputStream&);
void writeCbor(capnp::DynamicValue::Reader, kj::OutputStream&);

void writeJson(capnp::DynamicStruct::Reader, kj::OutputStream&, bool strict = true);
void writeJson(capnp::DynamicList::Reader, kj::OutputStream&, bool strict = true);
void writeJson(capnp::DynamicValue::Reader, kj::OutputStream&, bool strict = true);

}