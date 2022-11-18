#pragma once

#include "common.h"

# include <capnp/rpc-twoparty.h>

namespace fsc { namespace lvn {

using LocalVatNetworkBase = capnp::TwoPartyVatNetworkBase;
using LocalMessage = Own<capnp::MallocMessageBuilder>;

using Connection = LocalVatNetworkBase::Connection;

using VatId = capnp::rpc::twoparty::VatId;

Tuple<Own<Connection>, Own<Connection>> newLocalConnection();

}}