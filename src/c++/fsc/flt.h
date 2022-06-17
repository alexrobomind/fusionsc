#pragma once

#include "local.h"

#include <fsc/flt.capnp.h>

namespace fsc {
	FLT::Client newCpuTracer(LibraryThread& lt);
}