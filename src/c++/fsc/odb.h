#pragma once

#include <unordered_map>
#include <botan/hash.h>
#include <capnp/capability.h>

#include <fsc/warehouse.capnp.h>

#include "db.h"
#include "compression.h"

namespace fsc {
	
Warehouse::Client openWarehouse(db::Connection& conn, bool readOnly = false, kj::StringPtr tablePrefix = "warehouse");

}