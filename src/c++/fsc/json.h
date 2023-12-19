#pragma once

#include "common.h"

#include <capnp/dynamic.h>
#include <kj/io.h>

namespace fsc {

struct JsonOptions {
	enum Dialect {
		JSON, CBOR, BSON
	};
	
	Dialect dialect = JSON;
	
	bool quoteSpecialNums = true;
	kj::StringPtr jsonInf = "inf";
	kj::StringPtr jsonNegInf = "-.inf";
	kj::StringPtr jsonNan = ".nan";
	
	inline bool isBinary() { return dialect != JSON; }
};

void loadJson(capnp::DynamicStruct::Builder, kj::BufferedInputStream&, const JsonOptions& = JsonOptions());
void loadJson(capnp::ListSchema, kj::Function<capnp::DynamicList::Builder(size_t)>, kj::BufferedInputStream&, const JsonOptions& = JsonOptions());

capnp::DynamicValue::Reader loadJsonPrimitive(capnp::Type, kj::BufferedInputStream&, const JsonOptions& = JsonOptions());

void writeJson(capnp::DynamicValue::Reader, kj::BufferedOutputStream&, const JsonOptions& = JsonOptions());

}