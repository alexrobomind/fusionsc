#include "structio.h"

namespace fsc {
	void writeJsonSchema(capnp::Type, structio::Visitor&);
	void writeJsonSchema(capnp::Schema, structio::Visitor&);
}