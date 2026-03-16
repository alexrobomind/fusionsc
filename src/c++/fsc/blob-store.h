#pragma once

#include "db.h"

#include <kj/io.h>
#include <kj/async-io.h>

namespace fsc {

/**
 * \defgroup storage Storage
 *
 * \defgroup blobs Blob storage
 * @{
 * \brief Compressed hash-based database storage of binary data streams.
 * \ingroup storage
 * 
 * Hash-indexed binary data storage system. Blobs are written and read asynchronously. BLOB keys are created upon write completion.
 * 
 * \note
 * This system exists primarily to support the warehouse functionality. If you have other uses for this subsystems that you find
 * not well supported, please feel free to file an issue to have its interface adapted.
 *
 * Implementation note:
 *
 * The blob storage engine stores large binary objects as streamables inside an SQLite database. Streams are compress (ZLib) and
 * then broken down into small chunks that are written into the database. Keying the blobs on a hash ensures proper data dedupli-
 * cation.
 *
 * Currently, we use Botan's "Blake2b" hash function. 
 */

//! Extended input stream interface that can outsource the compression to an external thread. Inherits from kj::InputStream.
struct BlobReader : public kj::InputStream {
	virtual Promise<size_t> tryReadAsync(void* buf, size_t min, size_t max, const kj::Executor& decompressionThread) = 0;
};

//! Blob object
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
	
	//! Checks whether incRef may be called on this blob.
	inline bool isFinished() { return getHash() != nullptr; }
	
	//! Opens the blob for reading
	virtual Own<BlobReader> open() = 0;
};

/**
 * \brief Construction helper for new Blobs. Inherits from kj::OutputStream.
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
	/**
	 * Alternative write API. This does not call the underlying database and only manipulates
	 * the compressor and the buffer. It can be called from other thread as long as no simul
	 * taneous other calls to the blob builder are made.
	 *
	 * Returns true if the buffer was completely consumed. If it returns false, flush() must
	 * be called before calling tryConsume again. In this case, the next call to tryConsume
	 * MUST hold the same data as the previous call.
	 */
	virtual bool tryConsume(kj::ArrayPtr<const byte> input) = 0;
	
	//! Flushes buffer to database. UNLIKE tryConsume CAN NOT BE USED CROSS-THREAD
	virtual void flush() = 0;
	
	/**
	 * After write completion and hash key assignment, returns the finished blob (not
	 * neccessarily the same as the blob returned by getBlobUnderConstruction()
	 */
	virtual Own<Blob> finish() = 0;	
	
	virtual Own<Blob> getBlobUnderConstruction() = 0;
};

//! Blob storage interface.
struct BlobStore {
	virtual Own<BlobStore> addRef() = 0;
	
	//! Looks for a blob object in the database based on the given has key.
	virtual Maybe<Own<Blob>> find(kj::ArrayPtr<const byte> hash) = 0;
	
	//! Returns a blob by its database ID.
	virtual Own<Blob> get(int64_t id) = 0;
	
	//! Allocates a new blob without hash key in the database to be written to.
	virtual Own<BlobBuilder> create(size_t chunkSize) = 0;
};

Own<BlobStore> createBlobStore(db::Connection& conn, kj::StringPtr tablePrefix = "blobs", bool readOnly = false);

/**
 * @}
 */

}