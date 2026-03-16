#pragma once

#include "db.h"
#include "blob-store.h"
#include "data.h"

namespace fsc {

/**
 * \defgroup dbcache DB cache
 * @{
 *
 * \brief Temporary file system cache for DataRef objects.
 * \ingroup storage
 */

//! Database cache object.
struct DBCache {
	virtual Own<DBCache> addRef() = 0;
	
	//! Downloads DataRef into cache and returns stored object.
	virtual DataRef<capnp::AnyPointer>::Client cache(DataRef<capnp::AnyPointer>::Client) = 0;
	
	//! Downloads DataRef into cache and returns stored object.
	template<typename T>
	typename DataRef<T>::Client cache(typename DataRef<T>::Client);
	
	//! Downloads DataRef into cache and returns stored object.
	template<typename T>
	typename DataRef<T>::Client cache(LocalDataRef<T>);
};

//! Creates DB cache running atop existing fsc::BlobStore object.
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

 /** @} */

}