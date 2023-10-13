#include "blob-store.h"

namespace fsc {

namespace {

struct BlobStoreImpl : public BlobStore, kj::Refcounted {	
	BlobStoreImpl(sqlite::Connection& conn, kj::StringPtr tablePrefix, bool readOnly = false);
	
	Own<BlobStore> addRef() override;
	
	Maybe<Own<Blob>> find(kj::ArrayPtr<const byte> hash) override;
	Own<Blob> get(uint64_t id) override;
	
	Own<BlobBuilder> create(size_t chunkSize) override;
	
	// Impl
	using Statement = sqlite::Statement;
	
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
	Own<sqlite::Connection> conn;	
	const bool readOnly;
};

struct BlobImpl : public Blob, kj::Refcounted {
	BlobImpl(BlobStore& parent, uint64_t id);
	
	Own<Blob> addRef() override;
	
	void incRef() override;
	void decRef() override;
	
	int64_t getRefcount() override;
	kj::Array<const byte> getHash() override;
	int64_t getId() override;
	
	Own<kj::InputStream> open() override;
	
	// Impl
	Own<BlobStoreImpl> parent;
	uint64_t id;
};

struct BlobBuilderImpl : public BlobBuilder {
	BlobBuilderImpl(BlobStore& parent, size_t chunkSize);
	
	int64_t getId() override;
	void write(const void* buffer, size_t size) override;
	Own<Blob> finish() override;
	
	void writeBuffer();
		
	Own<BlobStore> parent;
	int64_t id;
	int64_t currentChunkNo = 0;
	kj::Array<byte> buffer;
	
	Compressor compressor;
	std::unique_ptr<Botan::HashFunction> hashFunction;
};

struct BlobReaderImpl : public kj::InputStream {
	BlobReaderImpl(BlobStore& parent, int64_t id);
	
	size_t tryRead(void* buf, size_t min, size_t max) override;
	
	Decompressor decompressor;
	sqlite::Statement readStatement;
	sqlite::Statement::Query readQuery;
};

// Implementation

// class BlobStoreImpl

BlobStoreImpl::BlobStoreImpl(sqlite::Connection& conn, kj::StringPtr tablePrefix, bool readOnly) :
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
	auto q = findBlob.query(hash);
	if(q.step()) {
		return get(q[0]);
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
	KJ_REQUIRE(!parent -> readOnly);
	parent -> incRefcount(id);
}

void BlobImpl::decRef() {
	KJ_REQUIRE(!parent -> readOnly);
	parent -> decRefcount(id);
}

void BlobImpl::getRefcount() {
	auto q = parent -> readRefcount.query(id);
	KJ_REQUIRE(q.step(), "Blob not found");
	
	return q[0].asInt64();
}

kj::Array<const byte> BlobImpl::getHash() {
	auto q = parent -> getBlobHash.query(id);
	
	if(!q.step())
		return nullptr;
	
	return kj::heapArray<byte>(q[0].asBlob());
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
	parent(parent.addRef()),
	buffer(kj::heapArray<byte>(chunkSize)),
	
	compressor(9),
	hashFunction(Botan::HashFunction::create("Blake2b"))
{
	KJ_REQUIRE(!parent.readOnly);
	KJ_REQUIRE(chunkSize > 0);
	compressor.setOutput(buffer);
}

void BlobBuilderImpl::write(void* buf, size_t count) {
	kj::ArrayPtr<const byte> data((const byte*) buf, count);
	auto t = parent -> conn -> ensureTransaction(true);
	
	hashFunction -> update(data.begin(), data.size());
	compressor.setInput(data);
	
	while(true) {
		compressor.step(false);
		
		if(compressor.remainingOut() == 0)
			flushBuffer();
		
		if(compressor.remainingIn() == 0)
			return;
	}
}

Own<Blob> BlobBuilderImpl::finish() {
	KJ_REQUIRE(buffer != nullptr, "Can only call BlobBuilder::finish() once");
	auto t = parent -> conn -> ensureTransaction(true);
	
	// Write out remaining data inside zlib stream
	compressor.setInput(nullptr);
	
	while(compressor.step(true) != ZLib::FINISHED) {
		flushBuffer();
	}
	
	flushBuffer();
	buffer = nullptr;
	
	// Finalize hash
	KJ_STACK_ARRAY(uint8_t, hashOutput, hashFunction -> output_length(), 1, 64);
	hashFunction -> final(hashOutput.begin());
	
	// Check hash for uniqueness
	KJ_IF_MAYBE(pBlob, parent -> find(hashOutput)) {
		return mv(*pBlob);
	}
	
	// Finalize hash and return blob
	parent -> setBlobHash(id, hashOutput);
	return kj::refcounted<BlobImpl>(*parent, id);
}

void BlobBuilderImpl::flushBuffer() {
	auto chunkData = buffer.slice(0, buffer.size() - compressor.remainingOut());
	
	if(chunkData.size() > 0) {
		parent -> createChunk(id, currentChunkNo, chunkData);
		++currentChunkNo;
	}
	compressor.setOutput(buffer);
}

// class BlobReaderImpl

BlobReaderImpl::BlobReaderImpl(BlobStoreImpl& parent, int64_t id) :
	parent(kj::addRef(parent)),
	id(id),
	readStatement(parent.conn -> prepare(str("SELECT data FROM ", parent.tablePrefix, "_chunks WHERE id = ? ORDER BY chunkNo"))),
	readQuery(readStatement.query(id))
{}

size_t BlobReaderImpl::tryRead(void* output, size_t minSize, size_t maxSize) {
	decompressor.setOutput(kj::ArrayPtr<byte>((byte)* output, maxSize));
	
	while(true) {
		if(decompressor.remainingIn() == 0) {
			KJ_REQUIRE(readQuery.step(), "Missing chunks despite expecting more");
			decompressor.setInput(readQuery[0]);
		}
			
		ZLib::State state = decompressor.step();
		
		if(state == ZLib::FINISHED)
			break;
		
		if(decompressor.remainingOut() == 0)
			break;
		
		KJ_ASSERT(decompressor.remainingIn() == 0);
	}
	
	return maxSize - decompressor.remainingOut();
}

}

Own<BlobStore> createBlobStore(sqlite::Connection& conn, kj::StringPtr tablePrefix, bool readOnly) {
	return kj::refcounted<BlobStoreImpl>(conn, tablePrefix, readOnly);
}

}