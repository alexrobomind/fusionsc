#pragma once

#include "capi-lvn.h"
#include "common.h"

#include <capnp/rpc.h>
#include <fsc/local-vat-network.capnp.h>

namespace fsc {

using LocalVatNetworkBase = capnp::VatNetwork<
	lvn::VatId,
	lvn::ProvisionId,
	lvn::RecipientId,
	lvn::ThirdPartyCapId,
	lvn::JoinResult
>;

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
	static lvn::VatId::Reader INITIAL_VAT_ID;
	virtual lvn::VatId::Reader getVatId() const = 0;
	
	virtual Maybe<Own<LocalVatNetworkBase::Connection>> connect(lvn::VatId::Reader hostId) override = 0;
	virtual Promise<kj::Own<LocalVatNetworkBase::Connection>> accept() override = 0;
};

/** Exchange hub for local vat networks.
 * 
 * Whereas the LocalVatNetwork class represents the endpoints in the local network,
 * the hub represents the network itself. Its join() method can be used to obtain new
 * endpoints in the network with unique addresses. joins will be performed sequentially,
 * and the first join() call is always guaranteed to have an ID of 0. Subsequent join()
 * calls give no promises about the used IDs. After a network is deallocated, its ID
 * will be reclaimed and might be used again. This inclused the initial 0 ID.
 *
 * This class actually wraps a C interface struct which can be passed to other fusionsc
 * instances without ABI incompatibility considerations. 
 */
struct LocalVatHub {
	LocalVatHub(fusionsc_LvnHub* = nullptr);
	
	LocalVatHub(const LocalVatHub&);
	LocalVatHub(LocalVatHub&&);
	
	~LocalVatHub();
	
	LocalVatHub& operator=(const LocalVatHub&);
	LocalVatHub& operator=(LocalVatHub&&);
	
	Own<LocalVatNetwork> join() const;
	
	fusionsc_LvnHub* incRef();
	fusionsc_LvnHub* release();
	
private:
	fusionsc_LvnHub* backend;
};

}