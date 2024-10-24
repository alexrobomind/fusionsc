#pragma once

#include "local.h"
#include "local-vat-network.h"

#include <fsc/services.capnp.h>

namespace fsc {

Own<RootService::Server> createRoot(LocalConfig::Reader config);
Own<LocalResources::Server> createLocalResources(LocalConfig::Reader config);

//! List of interface IDs that may not be called via network callss
kj::ArrayPtr<uint64_t> protectedInterfaces();
	
}
