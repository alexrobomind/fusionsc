#pragma once

#include "db.h"
#include "blob-store.h"
#include "data.h"

namespace fsc {

struct DBCache {
	virtual Own<DBCache> addRef() = 0;
	virtual DataRef<capnp::AnyPointer>::Client cache(DataRef<capnp::AnyPointer>::Client) = 0;
	
	template<typename T>
	typename DataRef<T>::Client cache(typename DataRef<T>::Client);
	
	template<typename T>
	typename DataRef<T>::Client cache(LocalDataRef<T>);
};

Own<DBCache> createDBCache(BlobStore& store);

// Inline implementation

template<typename T>
typename DataRef<T>::Client DBCache::cache(typename DataRef<T>::Client client) {
	return cache(client.asGeneric()).template asGeneric<T>();
}
	
template<typename T>
typename DataRef<T>::Client DBCache::cache(LocalDataRef<T> ldRef) {
	return cache<T>((typename DataRef<T>::Client&) ldRef);
}

}