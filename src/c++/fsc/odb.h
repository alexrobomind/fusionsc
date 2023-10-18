#pragma once

#include <unordered_map>
#include <botan/hash.h>
#include <capnp/capability.h>

#include <fsc/odb.capnp.h>

#include "db.h"
#include "compression.h"

namespace fsc {
	
odb::Folder::Client openObjectDb(db::Connection& conn, kj::StringPtr tablePrefix = "objectdb");

}