#pragma once

#include "db.h"
#include "blob-store.h"
#include "data.h"

namespace fsc {

struct DBCache {
	virtual Own<DBCache> addRef() = 0;
	virtual Promise<DataRef<capnp::AnyPointer>::Client> cache(DataRef<capnp::AnyPointer>::Client) = 0;
};

Own<DBCache> createCache(BlobStore& store);

}