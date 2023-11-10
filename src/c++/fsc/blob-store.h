#pragma once

#include "db.h"

#include <kj/io.h>

namespace fsc {
	
struct Blob {
	virtual Own<Blob> addRef() = 0;
	
	//! Increases the in-database refcount.
	virtual void incRef() = 0;
	
	//! Decreases the in-database refcount and deletes the blob if it hits 0.
	virtual void decRef() = 0;
	
	//! Reads the current in-database refcount
	virtual int64_t getRefcount() = 0;
	
	//! Reads the hash of the blob (or nullptr if the blob is under construction or deleted)
	virtual kj::Array<const byte> getHash() = 0;
	
	//! Returns the ID of the blob
	virtual int64_t getId() = 0;
	
	inline bool isFinished() { return getHash() != nullptr; }
	
	//! Opens the blob for reading
	virtual Own<kj::InputStream> open() = 0;
};

/** Construction helper for new Blobs
 *
 * This interface is responsible for managing the construction of a new Blob for the store.
 * It can only be obtained through BlobStore::create(). Upon creation, a blob under construction
 * is created with an implicit reference count of 1. The reference count may not be increased
 * until the blob's construction is finished.
 *
 * The blob may be filled with data through the OutputStream interface. Once all data have been
 * transferred, call the finish() method.
 *
 * finish() will check for hash duplication.If the new blob's hash is unique, that blob will be
 * returned (and from now on its refcount can be increased). In case of a hash conflict, the
 * new Blob will be deleted, the originally present blob with same hash will have its refcount
 * increased by 1, and that blob will be returned.
 *
 * This allows for interactions where the lifetime of the under-construction blob is attached
 * to an external object during the construction phase, and the Blob can be deleted implicitly
 * if that object gets deleted.
 */
struct BlobBuilder : public kj::OutputStream {	
	virtual Own<Blob> finish() = 0;	
	virtual Own<Blob> getBlobUnderConstruction() = 0;
};

struct BlobStore {
	virtual Own<BlobStore> addRef() = 0;
	
	virtual Maybe<Own<Blob>> find(kj::ArrayPtr<const byte> hash) = 0;
	virtual Own<Blob> get(int64_t id) = 0;
	
	virtual Own<BlobBuilder> create(size_t chunkSize) = 0;
};

Own<BlobStore> createBlobStore(db::Connection& conn, kj::StringPtr tablePrefix = "blobs", bool readOnly = false);

}