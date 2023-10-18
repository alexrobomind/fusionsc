#include "blob-store.h"
#include "compression.h"

#include <memory>
#include <botan/hash.h>

namespace fsc {

namespace {

struct BlobStoreImpl : public BlobStore, kj::Refcounted {	
	BlobStoreImpl(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly = false);
	
	Own<BlobStore> addRef() override;
	
	Maybe<Own<Blob>> find(kj::ArrayPtr<const byte> hash) override;
	Own<Blob> get(int64_t id) override;
	
	Own<BlobBuilder> create(size_t chunkSize) override;
	
	// Impl
	using Statement = db::PreparedStatement;
	
	Statement createBlob;
	Statement setBlobHash;
	Statement findBlob;
	Statement getBlobHash;
	
	Statement incRefcount;
	Statement decRefcount;
	Statement readRefcount;
	
	Statement deleteIfOrphan;
	Statement createChunk;
	
	kj::String tablePrefix;
	Own<db::Connection> conn;	
	const bool readOnly;
};

struct BlobImpl : public Blob, kj::Refcounted {
	BlobImpl(BlobStoreImpl& parent, int64_t id);
	
	Own<Blob> addRef() override;
	
	void incRef() override;
	void decRef() override;
	
	int64_t getRefcount() override;
	kj::Array<const byte> getHash() override;
	int64_t getId() override;
	
	Own<kj::InputStream> open() override;
	
	// Impl
	Own<BlobStoreImpl> parent;
	int64_t id;
};

struct BlobBuilderImpl : public BlobBuilder {
	BlobBuilderImpl(BlobStoreImpl& parent, size_t chunkSize);
	
	void write(const void* buffer, size_t size) override;
	void prepareFinish() override;
	Own<Blob> finish() override;
	Own<Blob> getBlobUnderConstruction() override;
	
	void flushBuffer();
		
	Own<BlobStoreImpl> parent;
	int64_t id;
	int64_t currentChunkNo = 0;
	kj::Array<byte> buffer;
	
	Compressor compressor;
	std::unique_ptr<Botan::HashFunction> hashFunction;
	
	Maybe<size_t> partialCompressionOffset;
	kj::Array<uint8_t> hash;
};

struct BlobReaderImpl : public kj::InputStream {
	BlobReaderImpl(BlobStoreImpl& parent, int64_t id);
	
	size_t tryRead(void* buf, size_t min, size_t max) override;
	
	Decompressor decompressor;
	db::PreparedStatement readStatement;
};

// Implementation

// class BlobStoreImpl

BlobStoreImpl::BlobStoreImpl(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly) :
	tablePrefix(kj::heapString(tablePrefix)),
	conn(conn.addRef()),
	readOnly(readOnly)
{
	if(!readOnly) {
		conn.exec(str(
			"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_blobs ("
			"  id INTEGER PRIMARY KEY,"
			"  hash BLOB UNIQUE," // SQLite UNIQUE allows multiple NULL values
			"  refcount INTEGER DEFAULT 1"
			")"
		));
		conn.exec(str(
			"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_chunks ("
			"  id INTEGER,"
			"  chunkNo INTEGER,"
			"  data BLOB,"
			""
			"  PRIMARY KEY(id, chunkNo),"
			"  FOREIGN KEY(id) REFERENCES ", tablePrefix, "_blobs(id) ON UPDATE CASCADE ON DELETE CASCADE"
			")"
		));
		conn.exec(str("CREATE INDEX IF NOT EXISTS ", tablePrefix, "_blobs_hash_idx ON ", tablePrefix, "_blobs (hash)"));
		conn.exec(str("CREATE UNIQUE INDEX IF NOT EXISTS ", tablePrefix, "_blobs_chunks_idx ON ", tablePrefix, "_chunks (id, chunkNo)"));
		
		createBlob = conn.prepare(str("INSERT INTO ", tablePrefix, "_blobs DEFAULT VALUES"));
		setBlobHash = conn.prepare(str("UPDATE ", tablePrefix, "_blobs SET hash = ?2 WHERE id = ?1"));
		
		incRefcount = conn.prepare(str("UPDATE ", tablePrefix, "_blobs SET refcount = refcount + 1 WHERE id = ?"));
		decRefcount = conn.prepare(str("UPDATE ", tablePrefix, "_blobs SET refcount = refcount - 1 WHERE id = ?"));
		
		deleteIfOrphan = conn.prepare(str("DELETE FROM ", tablePrefix, "_blobs WHERE id = ? AND refcount <= 0"));
		
		createChunk = conn.prepare(str("INSERT INTO ", tablePrefix, "_chunks (id, chunkNo, data) VALUES (?, ?, ?)"));
	}
	
	findBlob = conn.prepare(str("SELECT id FROM ", tablePrefix, "_blobs WHERE hash = ?"));
	getBlobHash = conn.prepare(str("SELECT hash FROM ", tablePrefix, "_blobs WHERE id = ?"));
	readRefcount = conn.prepare(str("SELECT refcount FROM ", tablePrefix, "_blobs WHERE id = ?"));
}

Own<BlobStore> BlobStoreImpl::addRef() {
	return kj::addRef(*this);
}

Maybe<Own<Blob>> BlobStoreImpl::find(kj::ArrayPtr<const byte> hash) {	
	findBlob.bind(hash);
	if(findBlob.step()) {
		return get(findBlob[0]);
	}
	
	return nullptr;
}

Own<Blob> BlobStoreImpl::get(int64_t id) {
	return kj::refcounted<BlobImpl>(*this, id);
}

Own<BlobBuilder> BlobStoreImpl::create(size_t chunkSize) {
	return kj::heap<BlobBuilderImpl>(*this, chunkSize);
}

// class BlobImpl

BlobImpl::BlobImpl(BlobStoreImpl& parent, int64_t id) :
	parent(kj::addRef(parent)),
	id(id)
{}

Own<Blob> BlobImpl::addRef() {
	return kj::addRef(*this);
}

void BlobImpl::incRef() {
	db::Transaction t(*parent -> conn);
	KJ_REQUIRE(!parent -> readOnly);
	KJ_REQUIRE(isFinished(), "Can not increase the reference count of deleted or under-construction blobs");
	parent -> incRefcount(id);
}

void BlobImpl::decRef() {
	db::Transaction t(*parent -> conn);
	KJ_REQUIRE(!parent -> readOnly);
	
	parent -> decRefcount(id);
	parent -> deleteIfOrphan(id);
}

int64_t BlobImpl::getRefcount() {
	auto& rc = parent -> readRefcount;
	
	rc.bind(id);
	KJ_REQUIRE(rc.step(), "Blob not found");
	
	return rc[0];
}

kj::Array<const byte> BlobImpl::getHash() {
	auto& gbh = parent -> getBlobHash;
	
	gbh.bind(id);
	
	if(!gbh.step())
		return nullptr;
	
	return kj::heapArray<byte>(gbh[0].asBlob());
}

int64_t BlobImpl::getId() {
	return id;
}

Own<kj::InputStream> BlobImpl::open() {
	KJ_REQUIRE(isFinished(), "Can not open an unfinished blob");
	return kj::heap<BlobReaderImpl>(*parent, id);
}

// class BlobBuilderImpl

BlobBuilderImpl::BlobBuilderImpl(BlobStoreImpl& parent, size_t chunkSize) :
	id(parent.createBlob.insert()),
	parent(kj::addRef(parent)),
	buffer(kj::heapArray<byte>(chunkSize)),
	
	compressor(9),
	hashFunction(Botan::HashFunction::create("Blake2b"))
{
	KJ_REQUIRE(!parent.readOnly);
	KJ_REQUIRE(chunkSize > 0);
	compressor.setOutput(buffer);
}

void BlobBuilderImpl::write(const void* buf, size_t count) {
	KJ_REQUIRE(!parent -> conn -> inTransaction(), "Can not call write() inside transaction");
	
	kj::ArrayPtr<const byte> data((const byte*) buf, count);
	
	// If a previous run failed, pick up where we left off
	KJ_IF_MAYBE(pOffset, partialCompressionOffset) {
		compressor.setInput(data.slice(*pOffset, data.size()));
	} else {
		compressor.setInput(data);
	}
	
	while(true) {
		compressor.step(false);
		
		if(compressor.remainingOut() == 0) {
			partialCompressionOffset = data.size() - compressor.remainingIn();
			flushBuffer();
			partialCompressionOffset = nullptr;
		}
		
		if(compressor.remainingIn() == 0)
			break;
	}
	
	hashFunction -> update(data.begin(), data.size());
}

Own<Blob> BlobBuilderImpl::getBlobUnderConstruction() {
	KJ_REQUIRE(buffer != nullptr, "Can only access partially constructed blob before finish()");
	return kj::refcounted<BlobImpl>(*parent, id);
}

void BlobBuilderImpl::prepareFinish() {
	KJ_REQUIRE(!parent -> conn -> inTransaction(), "Can not call prepareFinish() inside transaction");
	if(buffer == nullptr)
		return;
	
	// Write out remaining data inside zlib stream
	compressor.setInput(nullptr);
	
	while(compressor.step(true) != ZLib::FINISHED) {
		flushBuffer();
	}
	
	flushBuffer();
	buffer = nullptr;
	
	hash = kj::heapArray<uint8_t>(hashFunction -> output_length());
	hashFunction -> final(hash.begin());
}

Own<Blob> BlobBuilderImpl::finish() {
	KJ_REQUIRE(hash != nullptr, "Must call prepareFinish() before calling finish()");
	KJ_REQUIRE(parent -> conn -> inTransaction(), "finish() must be called inside transaction");
	
	// Check hash for uniqueness
	KJ_IF_MAYBE(pBlob, parent -> find(hash)) {
		(**pBlob).incRef();
		BlobImpl(*parent, id).decRef();
		return mv(*pBlob);
	}
	
	// Finalize hash and return blob
	parent -> setBlobHash(id, hash.asPtr());
	return kj::refcounted<BlobImpl>(*parent, id);
}

void BlobBuilderImpl::flushBuffer() {
	db::Transaction t(*parent -> conn);
	{
		auto chunkData = buffer.slice(0, buffer.size() - compressor.remainingOut());
		
		if(chunkData.size() > 0) {
			parent -> createChunk(id, currentChunkNo, chunkData);
			++currentChunkNo;
		}
	}
	compressor.setOutput(buffer);
}

// class BlobReaderImpl

BlobReaderImpl::BlobReaderImpl(BlobStoreImpl& parent, int64_t id) :
	readStatement(parent.conn -> prepare(str("SELECT data FROM ", parent.tablePrefix, "_chunks WHERE id = ? ORDER BY chunkNo")))
{
	readStatement.bind(id);
}

size_t BlobReaderImpl::tryRead(void* output, size_t minSize, size_t maxSize) {
	decompressor.setOutput(kj::ArrayPtr<byte>((byte*) output, maxSize));
	
	while(true) {
		ZLib::State state = decompressor.step();
		
		if(state == ZLib::FINISHED)
			break;
		
		if(decompressor.remainingOut() == 0)
			break;
		
		KJ_ASSERT(decompressor.remainingIn() == 0);
		KJ_REQUIRE(readStatement.step(), "Missing chunks despite expecting more");
		decompressor.setInput(readStatement[0]);		
	}
	
	return maxSize - decompressor.remainingOut();
}

}

Own<BlobStore> createBlobStore(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly) {
	return kj::refcounted<BlobStoreImpl>(conn, tablePrefix, readOnly);
}

}