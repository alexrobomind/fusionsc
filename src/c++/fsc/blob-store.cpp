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

struct BlobImpl final : public Blob, kj::Refcounted {
	BlobImpl(BlobStoreImpl& parent, int64_t id);
	
	Own<Blob> addRef() override;
	
	void incRef() override;
	void decRef() override;
	
	int64_t getRefcount() override;
	kj::Array<const byte> getHash() override;
	int64_t getId() override;
	
	Own<BlobReader> open() override;
	
	// Impl
	Own<BlobStoreImpl> parent;
	int64_t id;
};

struct BlobBuilderImpl final : public BlobBuilder {
	BlobBuilderImpl(BlobStoreImpl& parent, size_t chunkSize, int compressionLevel);
	
	void write(const void* buffer, size_t size) override;
	Own<Blob> finish() override;
	Own<Blob> getBlobUnderConstruction() override;
	
	bool tryConsume(ArrayPtr<const byte>) override;
	void flush() override;
	
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

struct BlobReaderImpl final : public BlobReader {
	BlobReaderImpl(BlobStoreImpl& parent, int64_t id);
	
	size_t tryRead(void* buf, size_t min, size_t max) override;
	Promise<size_t> tryReadAsync(void* buf, size_t min, size_t max, const kj::Executor& decompressionThread) override;
	
	Promise<size_t> readAsyncStep(size_t min, size_t max, Own<const kj::Executor>);
	
	Decompressor decompressor;
	db::PreparedStatement readStatement;
	db::PreparedStatement::Query query;
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
			"  id INTEGER PRIMARY KEY AUTOINCREMENT,"
			"  hash BLOB UNIQUE DEFAULT NULL," // SQLite UNIQUE allows multiple NULL values
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
	auto q = findBlob.bind(hash);
	if(q.step()) {
		return get(q[0]);
	}
	
	return nullptr;
}

Own<Blob> BlobStoreImpl::get(int64_t id) {
	return kj::refcounted<BlobImpl>(*this, id);
}

Own<BlobBuilder> BlobStoreImpl::create(size_t chunkSize) {
	return kj::heap<BlobBuilderImpl>(*this, chunkSize, -1);
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
	
	auto q = rc.bind(id);
	KJ_REQUIRE(q.step(), "Blob not found");
	
	return q[0];
}

kj::Array<const byte> BlobImpl::getHash() {
	auto& gbh = parent -> getBlobHash;
	
	auto q = gbh.bind(id);
	KJ_REQUIRE(q.step(), "Blob not found");
		
	return kj::heapArray<byte>(q[0].asBlob());
}

int64_t BlobImpl::getId() {
	return id;
}

Own<BlobReader> BlobImpl::open() {
	KJ_REQUIRE(isFinished(), "Can not open an unfinished blob");
	return kj::heap<BlobReaderImpl>(*parent, id);
}

// class BlobBuilderImpl

BlobBuilderImpl::BlobBuilderImpl(BlobStoreImpl& parent, size_t chunkSize, int compressionLevel) :
	id(parent.createBlob.insert()),
	parent(kj::addRef(parent)),
	buffer(kj::heapArray<byte>(chunkSize)),
	
	compressor(compressionLevel),
	hashFunction(Botan::HashFunction::create("Blake2b"))
{
	KJ_REQUIRE(!parent.readOnly);
	KJ_REQUIRE(chunkSize > 0);
	compressor.setOutput(buffer);
}

void BlobBuilderImpl::write(const void* buf, size_t count) {	
	// !! This must be restartable in case the first
	// !! flushBuffer call fails.
	db::Transaction transaction(*parent -> conn);
	
	kj::ArrayPtr<const byte> data((const byte*) buf, count);
	
	/*// If a previous run failed, pick up where we left off
	KJ_IF_MAYBE(pOffset, partialCompressionOffset) {
		KJ_REQUIRE(compressor.remainingIn() + partialCompressionOffset == count);
		// compressor.setInput(data.slice(*pOffset, data.size()));
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
	
	hashFunction -> update(data.begin(), data.size());*/
	while(!tryConsume(data)) {
		flush();
	}
}

bool BlobBuilderImpl::tryConsume(kj::ArrayPtr<const byte> data) {
	// If a previous run failed, pick up where we left off
	KJ_IF_MAYBE(pOffset, partialCompressionOffset) {
		KJ_REQUIRE(compressor.remainingIn() + *pOffset == data.size());
		// compressor.setInput(data.slice(*pOffset, data.size()));
	} else {
		compressor.setInput(data);
	}
	
	compressor.step(false);
	
	if(compressor.remainingIn() > 0) {
		partialCompressionOffset = data.size() - compressor.remainingIn();
		return false;
	}
	
	partialCompressionOffset = nullptr;
	hashFunction -> update(data.begin(), data.size());
	return true;
}

void BlobBuilderImpl::flush() {
	if(compressor.remainingOut() == 0)
		flushBuffer();
}

Own<Blob> BlobBuilderImpl::getBlobUnderConstruction() {
	KJ_REQUIRE(buffer != nullptr, "Can only access partially constructed blob before finish()");
	return kj::refcounted<BlobImpl>(*parent, id);
}

Own<Blob> BlobBuilderImpl::finish() {
	db::Transaction transaction(*parent -> conn);
	
	// !! This must be restartable in case the first
	// !! flushBuffer call fails or setting the hash fails
	if(buffer != nullptr) {	
		// Write out remaining data inside zlib stream
		compressor.setInput(nullptr);
		
		while(compressor.step(true) != ZLib::FINISHED) {
			flushBuffer();
		}
		
		flushBuffer();
		buffer = nullptr;
	}
	
	if(hash == nullptr) {	
		hash = kj::heapArray<uint8_t>(hashFunction -> output_length());
		hashFunction -> final(hash.begin());
	}
	KJ_REQUIRE(hash != nullptr, "Must call prepareFinish() before calling finish()");
	
	auto gbh = parent -> getBlobHash.bind(id);
	KJ_REQUIRE(gbh.step(), "finish() was already called and blob under construction was deleted");
	KJ_REQUIRE(gbh[0].isNull(), "finish() was already called and blob has hash assigned");
	
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
	readStatement(parent.conn -> prepare(str("SELECT data FROM ", parent.tablePrefix, "_chunks WHERE id = ? ORDER BY chunkNo"))),
	query(readStatement.bind(id))
{
}

size_t BlobReaderImpl::tryRead(void* output, size_t minSize, size_t maxSize) {
	KJ_ASSERT(maxSize >= minSize);
	
	decompressor.setOutput(kj::ArrayPtr<byte>((byte*) output, maxSize));
	
	while(true) {
		ZLib::State state = decompressor.step();
		size_t filled = maxSize - decompressor.remainingOut();
		
		if(state == ZLib::FINISHED || filled >= minSize)
			return filled;
		
		KJ_ASSERT(decompressor.remainingIn() == 0);
		KJ_REQUIRE(query.step(), "Missing chunks despite expecting more");
		decompressor.setInput(query[0]);		
	}
	
	// return maxSize - decompressor.remainingOut();
	KJ_UNREACHABLE;
}

Promise<size_t> BlobReaderImpl::tryReadAsync(void* output, size_t minSize, size_t maxSize, const kj::Executor& executor) {
	KJ_ASSERT(maxSize >= minSize);
	decompressor.setOutput(kj::ArrayPtr<byte>((byte*) output, maxSize));
	return readAsyncStep(minSize, maxSize, executor.addRef());
}

Promise<size_t> BlobReaderImpl::readAsyncStep(size_t minSize, size_t maxSize, Own<const kj::Executor> executor) {	
	return executor -> executeAsync([this]() {
		return decompressor.step();
	})
	.then([this, minSize, maxSize, e = executor -> addRef()](ZLib::State state) mutable -> Promise<size_t> {
		size_t filled = maxSize - decompressor.remainingOut();
		if(filled >= minSize || state == ZLib::FINISHED)
			return maxSize - decompressor.remainingOut();
		
		KJ_ASSERT(decompressor.remainingIn() == 0);
		KJ_REQUIRE(query.step(), "Missing chunks despite expecting more");
		decompressor.setInput(query[0]);
		
		return readAsyncStep(minSize, maxSize, mv(e));
	});
}

}

Own<BlobStore> createBlobStore(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly) {
	return kj::refcounted<BlobStoreImpl>(conn, tablePrefix, readOnly);
}

}