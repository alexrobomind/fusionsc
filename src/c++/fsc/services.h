#pragma once

#include "local.h"

#include <fsc/services.capnp.h>

namespace fsc {

kj::Function<capnp::Capability::Client()> newInProcessServer(kj::Function<capnp::Capability::Client()> serviceFactory);

template<typename T>
kj::Function<typename T::Client()> newInProcessServer(kj::Function<typename T::Client()> factory) {
	auto backend = newInProcessServer([factory = mv(factory)]() mutable -> capnp::Capability::Client { return factory(); });
	
	return [backend = mv(backend)]() mutable {
		return backend().template castAs<T>();
	};
}

Own<RootService::Server> createRoot(LocalConfig::Reader config);
Own<LocalResources::Server> createLocalResources(LocalConfig::Reader config);

RootService::Client connectRemote(kj::StringPtr address, unsigned int portHint = 0);

struct Server {
	virtual Promise<void> run() = 0;
	virtual Promise<void> drain() = 0;
	
	virtual unsigned int getPort() = 0;
	
	virtual ~Server() {};
};

Promise<Own<Server>> startServer(unsigned int portHint = 0, kj::StringPtr address = "0.0.0.0"_kj);

inline constexpr kj::StringPtr MAGIC_TOKEN = "I am an FSC server"_kj;

//! List of interface IDs that may not be called via network callss
kj::ArrayPtr<uint64_t> protectedInterfaces();
	
}