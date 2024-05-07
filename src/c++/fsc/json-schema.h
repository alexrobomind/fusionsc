#include "textio.h"

namespace fsc {
	void writeJsonSchema(capnp::Type, textio::Visitor&);
	void writeJsonSchema(capnp::Schema, textio::Visitor&);
}