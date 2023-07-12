#include "fscpy.h"
#include "async.h"

#include <fsc/data.h>
#include <fsc/services.h>

namespace fscpy {
	
capnp::Type getRefPayload(capnp::InterfaceSchema refSchema);
capnp::InterfaceSchema createRefSchema(capnp::Type payloadType);

Maybe<capnp::Type> getPayloadType(LocalDataRef<> dataRef);
	
kj::Own<PyObjectHolder> openRef(capnp::Type payloadType, LocalDataRef<> dataRef);
	
PyPromise download(capnp::DynamicCapability::Client capability);

capnp::DynamicCapability::Client publishReader(capnp::DynamicStruct::Reader value);
capnp::DynamicCapability::Client publishBuilder(capnp::DynamicStruct::Builder dsb);

}