#pragma once

#include "db.h"

#include <kj/io.h>

namespace fsc {
	
struct Blob {
	virtual Own<Blob> addRef() = 0;
	
	virtual void incRef() = 0;
	virtual void decRef() = 0;
	
	virtual int64_t getRefcount() = 0;
	virtual kj::Array<const byte> getHash() = 0;
	virtual int64_t getId() = 0;
	
	inline bool isFinished() { return getHash() != nullptr; }
	
	virtual Own<kj::InputStream> open() = 0;
};

struct BlobBuilder : public kj::OutputStream {
	virtual int64_t getId() = 0;
	virtual Own<Blob> finish() = 0;
};

struct BlobStore {
	virtual Own<BlobStore> addRef() = 0;
	
	virtual Maybe<Own<Blob>> find(kj::ArrayPtr<const byte> hash) = 0;
	virtual Own<Blob> get(int64_t id) = 0;
	
	virtual Own<BlobBuilder> create(size_t chunkSize) = 0;
};

Own<BlobStore> createBlobStore(sqlite::Connection& conn, kj::StringPtr tablePrefix, bool readOnly = false);

}