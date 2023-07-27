#pragma once

#include "fscpy.h"

namespace fscpy {

py::object pickleReduceReader(DynamicStructReader reader, uint32_t pickleVersion);
py::object pickleReduceBuilder(DynamicStructBuilder reader, uint32_t pickleVersion);
py::object pickleReduceRef(capnp::DynamicCapability::Client clt, uint32_t pickleVersion);

DynamicValueReader unpickleReader(uint32_t version, py::list data);
DynamicValueBuilder unpickleBuilder(uint32_t version, py::list data);
DynamicCapabilityClient unpickleRef(uint32_t version, py::list data);

}