#pragma once

#include "local.h"

#include <fsc/services.capnp.h>

namespace fsc {

RootService::Client createRoot(LibraryThread& lt, RootConfig::Reader config);

ResolverChain::Client newResolverChain();

RootService::Client connectRemote(LibraryThread& lt, kj::StringPtr address, unsigned int portHint = 0);

Promise<Tuple<unsigned int, Promise<void>>> startServer(LibraryThread& lt, unsigned int portHint = 0, kj::StringPtr address = "0.0.0.0"_kj);

inline constexpr kj::StringPtr MAGIC_TOKEN = "I am an FSC server"_kj;
	
}