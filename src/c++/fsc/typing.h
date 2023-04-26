#pragma once

#include <capnp/schema.capnp.h>
#include <capnp/schema.h>

namespace fsc {
	void extractBrand(capnp::Schema in, capnp::schema::Brand::Builder out);
	void extractType(capnp::Type in, capnp::schema::Type::Builder out);
}