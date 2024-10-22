#pragma once

#include "local.h"
#include "local-vat-network.h"

#include <fsc/services.capnp.h>

namespace fsc {

capnp::Capability::Client connectInProcess(const LocalVatHub&, uint64_t address = 0);

struct InProcessServer {
	virtual LocalVatHub getHub() const = 0;
	
	virtual Own<const InProcessServer> addRef() const = 0;
	
	template<typename T>
	typename T::Client connect() const {
		return connectBase().castAs<T>();
	}
	
	inline capnp::Capability::Client connectBase() const {
		return connectInProcess(getHub());
	}
};

Own<const InProcessServer> newInProcessServer(kj::Function<capnp::Capability::Client()> serviceFactory, Library contextLibrary = getActiveThread().library()->addRef());

Own<RootService::Server> createRoot(LocalConfig::Reader config);
Own<LocalResources::Server> createLocalResources(LocalConfig::Reader config);

// RootService::Client connectRemote(kj::StringPtr address, unsigned int portHint = 0);

struct Server {
	virtual Promise<void> run() = 0;
	virtual Promise<void> drain() = 0;
	
	virtual unsigned int getPort() = 0;
	
	virtual ~Server() {};
};

// Promise<Own<Server>> startServer(unsigned int portHint = 0, kj::StringPtr address = "0.0.0.0"_kj);

inline constexpr kj::StringPtr MAGIC_TOKEN = "I am an FSC server"_kj;

//! List of interface IDs that may not be called via network callss
kj::ArrayPtr<uint64_t> protectedInterfaces();
	
}
