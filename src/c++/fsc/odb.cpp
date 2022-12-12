#include "odb.h"

using kj::str;

namespace fsc {
	
BlobStore::BlobStore(sqlite::Connection& connRef, kj::StringPtr tablePrefix) :
	tablePrefix(kj::heapString(tablePrefix)),
	conn(connRef.addRef())
{
	connRef.exec(str(
		"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_blobs ("
		"  id INTEGER PRIMARY KEY,"
		"  hash BLOB UNIQUE," // SQLite UNIQUE allows multiple NULL values
		"  externalRefcount INTEGER,"
		"  internalRefcount INTEGER"
		")"
	));
	connRef.exec(str(
		"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_chunks ("
		"  id INTEGER REFERENCES ", tablePrefix, "_blobs(id) ON UPDATE CASCADE ON DELETE CASCADE,"
		"  chunkNo INTEGER,"
		"  data BLOB,"
		""
		"  PRIMARY KEY(id, chunkNo)"
		")"
	));
	connRef.exec(str("CREATE INDEX IF NOT EXISTS ", tablePrefix, "_blobs_hash_idx ON ", tablePrefix, "_blobs (hash)"));
	connRef.exec(str("CREATE UNIQUE INDEX IF NOT EXISTS ", tablePrefix, "_blobs_chunks_idx ON ", tablePrefix, "_chunks (id, chunkNo)"));
		
	createBlob = conn->prepare(str("INSERT INTO ", tablePrefix, "_blobs DEFAULT VALUES"));
	setBlobHash = conn->prepare(str("UPDATE ", tablePrefix, "_blobs SET hash = ?2 WHERE id = ?1"));
	findBlob = conn->prepare(str("SELECT id FROM ", tablePrefix, "_blobs WHERE hash = ?"));
	
	incRefExternal = conn->prepare(str("UPDATE ", tablePrefix, "_blobs SET externalRefcount = externalRefcount + 1 WHERE id = ?"));
	decRefExternal = conn->prepare(str("UPDATE ", tablePrefix, "_blobs SET externalRefcount = externalRefcount - 1 WHERE id = ?"));
	incRefInternal = conn->prepare(str("UPDATE ", tablePrefix, "_blobs SET internalRefcount = internalRefcount + 1 WHERE id = ?"));
	decRefInternal = conn->prepare(str("UPDATE ", tablePrefix, "_blobs SET internalRefcount = internalRefcount - 1 WHERE id = ?"));
	
	deleteIfOrphan = conn->prepare(str("DELETE FROM ", tablePrefix, "_blobs WHERE id = ? AND externalRefcount = 0 AND internalRefcount = 0"));
	
	createChunk = conn->prepare(str("INSERT INTO ", tablePrefix, "_chunks (id, chunkNo, data) VALUES (?, ?, ?)"));
}

Maybe<Blob> BlobStore::find(kj::ArrayPtr<const byte> hash) {
	findBlob.bind(hash);
	KJ_DEFER({ findBlob.reset(); });
	
	if(findBlob.step()) {
		return Blob(*this, findBlob[0]);
	}
	
	return nullptr;
}

BlobBuilder BlobStore::create(size_t chunkSize) {
	return BlobBuilder(*this, chunkSize);
}

// =================================== class Blob ===================================

Blob::Blob(BlobStore& parent, int64_t id) :
	parent(parent.addRef()),
	id(id)
{
	parent.incRefInternal(id);
}

Blob::~Blob() {
	if(parent.get() == nullptr)
		return;
	
	ud.catchExceptionsIfUnwinding([this]() {
		parent -> decRefInternal(id);
		parent -> deleteIfOrphan(id);
	});
}

// ============================== class BlobBuilder =================================

BlobBuilder::BlobBuilder(BlobStore& parent, size_t chunkSize) :
	id(parent.createBlob.insert()),
	parent(parent.addRef()),
	buffer(kj::heapArray<byte>(chunkSize)),
	
	compressor(9),
	hashFunction(Botan::HashFunction::create("Blake2b"))
{
	KJ_REQUIRE(chunkSize > 0);
	compressor.setOutput(buffer);
}

BlobBuilder::~BlobBuilder() {
	ud.catchExceptionsIfUnwinding([this]() {
		parent-> deleteIfOrphan(id);
	});
}

void BlobBuilder::flushBuffer() {
	KJ_DBG("Flushing buffer");
	auto chunkData = buffer.slice(0, buffer.size() - compressor.remainingOut());
	
	if(chunkData.size() > 0) {
		parent -> createChunk(id, currentChunkNo, chunkData);
		++currentChunkNo;
	}
	compressor.setOutput(buffer);
}

void BlobBuilder::write(kj::ArrayPtr<const byte> data) {
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

Blob BlobBuilder::finish() {
	KJ_REQUIRE(buffer != nullptr, "Can only call BlobBuilder::finish() once");
	
	compressor.setInput(nullptr);
	
	while(compressor.step(true) != ZLib::FINISHED) {
		flushBuffer();
	}
	
	flushBuffer();
	buffer = nullptr;
		
	KJ_STACK_ARRAY(uint8_t, hashOutput, hashFunction -> output_length(), 1, 64);
	hashFunction -> final(hashOutput.begin());
	
	// We need to check for uniqueness of the target object. If the hash already exists, we return that object instead (this one will be deleted when the blob builder gets destroyed)
	auto transaction = parent -> conn -> beginTransaction();
	
	auto& findBlob = parent -> findBlob;
	
	KJ_IF_MAYBE(pBlob, parent -> find(hashOutput)) {
		return mv(*pBlob);
	}
	
	parent -> setBlobHash(id, hashOutput);
	return Blob(*parent, id);
}

// ============================================ class BlobReader ==============================================

BlobReader::BlobReader(Blob& blob) :
	blob(blob),
	readStatement(sqlite::Statement(blob.parent -> conn -> prepare(str("SELECT data FROM ", blob.parent -> tablePrefix, "_chunks WHERE id = ? ORDER BY chunkNo"))))
{
	readStatement.bind(blob.id);
}

bool BlobReader::read(kj::ArrayPtr<byte> output) {
	decompressor.setOutput(output);
	
	while(true) {
		if(decompressor.remainingIn() == 0) {
			KJ_REQUIRE(readStatement.step(), "Missing chunks despite expecting more");
			decompressor.setInput(readStatement[0]);
		}
			
		ZLib::State state = decompressor.step();
		
		if(state == ZLib::FINISHED)
			return true;
		
		if(decompressor.remainingOut() == 0)
			return false;
		
		KJ_ASSERT(decompressor.remainingIn() == 0);
	}
}

}