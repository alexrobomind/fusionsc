#pragma once

#include "local.h"

#include <fsc/services.capnp.h>

namespace fsc {

RootService::Client createRoot(LibraryThread& lt, RootConfig::Reader config);

ResolverChain::Client newResolverChain();

RootService::Client connectRemote(LibraryThread& lt, kj::StringPtr address, unsigned int portHint = 0);

struct Server {
	virtual Promise<void> run() = 0;
	virtual Promise<void> drain() = 0;
	
	virtual unsigned int getPort() = 0;
	
	virtual ~Server() {};
};

Promise<Own<Server>> startServer(LibraryThread& lt, unsigned int portHint = 0, kj::StringPtr address = "0.0.0.0"_kj);

inline constexpr kj::StringPtr MAGIC_TOKEN = "I am an FSC server"_kj;
	
}