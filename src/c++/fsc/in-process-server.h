#pragma once

#include "local-vat-network.h"
#include "local.h"

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

}
