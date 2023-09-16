#pragma once

#include "common.h"
#include "data.h"
#include "capi-lvn.h"

#include <capnp/rpc.h>
#include <fsc/local-vat-network.capnp.h>

#include <atomic>
#include <type_traits>

namespace fsc { 

using LocalVatNetworkBase = capnp::VatNetwork<
	lvn::VatId,
	lvn::ProvisionId,
	lvn::RecipientId,
	lvn::ThirdPartyCapId,
	lvn::JoinResult
>;

struct LocalVatNetwork;

//! Central exchange point to set up connections between LocalVatNetwork instances
struct LocalVatHub : public kj::AtomicRefcounted {
	//! Returns the Vat ID guaranteed to be assigned to the first vat joining the hub
	static lvn::VatId::Reader INITIAL_VAT_ID;
	
	Own<const LocalVatHub> addRef() const;
	
private:
	struct SharedData {
		kj::HashMap<uint64_t, const LocalVatNetwork*> vats;
		uint64_t freeId = 0;
	};
	kj::MutexGuarded<SharedData> data;
	
	friend class LocalVatNetwork;
};

Own<LocalVatHub> newLocalVatHub();

/** Local Vat network implementation
 *
 * Local multi-point implementation of Cap'n'proto's vat network protocol.
 * All LocalVatNetwork instances connected to the same LocalVatHub instance
 * can form connections with each other.
 *
 * \note An instance of LocalVatNetwork may not be simultaneously used from
 *       multiple threads. However, multiple threads can connect through the
 *       same LocalVatHub, and may form connections with each other.
 *       Additionally, it is also allowed to create a LocalVatNetwork instance
 *       in one thread but use it in another, as long as no method except
 *       getVatId() is called beforehand.
 */
struct LocalVatNetwork : public LocalVatNetworkBase {	
	//! Vat ID that other LocalVatNetwork instances on the same hub can use to connect to this instance
	lvn::VatId::Reader getVatId() const;
	
	Maybe<Own<LocalVatNetworkBase::Connection>> connect(lvn::VatId::Reader hostId) override;
	Promise<kj::Own<LocalVatNetworkBase::Connection>> accept() override;
		
	LocalVatNetwork(const LocalVatHub& hub);
	~LocalVatNetwork();
	
private:
	struct Impl;
	Own<Impl> pImpl;
};

}