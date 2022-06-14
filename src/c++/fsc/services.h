#pragma once

#include "local.h"

#include <fsc/services.capnp.h>

namespace fsc {

RootService::Client createRoot(LibraryThread& lt, RootConfig::Reader config);

ResolverChain::Client newResolverChain();
	
}