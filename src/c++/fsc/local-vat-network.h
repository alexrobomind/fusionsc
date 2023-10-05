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

struct LvnHub;
struct LocalVatNetwork;

struct LvnHub {
	LvnHub(fusionsc_LvnHub*);
	
	LvnHub(const LvnHub&);
	LvnHub(LvnHub&&);
	
	LvnHub& operator=(const LvnHub&);
	LvnHub& operator=(LvnHub&&);
	
	fusionsc_LvnHub* incRef() const;
	fusionsc_LvnHub* release() const;
	fusionsc_LvnHub* get() const;
	
private:
	fusionsc_LvnHub* backend;
};

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
		
	LocalVatNetwork(LvnHub& hub);
	~LocalVatNetwork();
	
private:
	struct Impl;
	Own<Impl> pImpl;
};

}