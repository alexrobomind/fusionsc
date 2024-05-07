#pragma once

#include "fscpy.h"
#include <fsc/dynamic.capnp.h>

namespace fscpy {
	py::object loadStructArray(WithMessage<DynamicObject::StructArray::Reader> structArray);
	py::object loadEnumArray(WithMessage<DynamicObject::EnumArray::Reader> structArray);
}