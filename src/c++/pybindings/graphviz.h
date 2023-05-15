#pragma once

#include "fscpy.h"

namespace fscpy {

py::object visualizeGraph(capnp::DynamicStruct::Reader reader, py::kwargs kwargs);

}