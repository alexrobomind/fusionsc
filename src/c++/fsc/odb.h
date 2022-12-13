#pragma once

#include <botan/hash.h>

#include "db.h"
#include "compression.h"

namespace fsc {
	
struct BlobStore;
struct Blob;
struct BlobReader;
struct BlobBuilder;

struct BlobStore : public kj::Refcounted {
	using Statement = sqlite::Statement;
	
	Statement createBlob;
	Statement setBlobHash;
	Statement findBlob;
	
	Statement incRefExternal;
	Statement decRefExternal;
	Statement incRefInternal;
	Statement decRefInternal;
	
	Statement deleteIfOrphan;
	Statement createChunk;
	
	Statement savepoint;
	Statement release;
	
	kj::String tablePrefix;
	Own<sqlite::Connection> conn;	

	inline Own<BlobStore> addRef() { return kj::addRef(*this); }
	
	Maybe<Blob> find(kj::ArrayPtr<const byte> hash);
	BlobBuilder create(size_t chunkSize);
		
private:
	BlobStore(sqlite::Connection& conn, kj::StringPtr tablePrefix);
	
	friend kj::Refcounted;
	
	template<typename T, typename... Params>
	friend Own<T> kj::refcounted(Params&&... params);
	
	template <typename T>
	friend Own<T> kj::addRef(T& object);	
};

struct Blob {
	mutable Own<BlobStore> parent;
	const int64_t id;
	
	Blob(BlobStore& parent, int64_t id);
	
	inline Blob(const Blob& other) : Blob(*(other.parent), other.id) {}
	inline Blob(Blob&& other) = default;
	
	inline BlobReader open();
	~Blob();
	
private:
	kj::UnwindDetector ud;
};

struct BlobBuilder {
	BlobBuilder(BlobStore& parent, size_t chunkSize = 8 * 1024 * 1024);
	~BlobBuilder();
	
	void write(kj::ArrayPtr<const byte> bytes);
	Blob finish();
	
private:
	int64_t id;
	int64_t currentChunkNo = 0;
	
	Own<BlobStore> parent;
	kj::Array<byte> buffer;
	
	void flushBuffer();
	
	Compressor compressor;
	std::unique_ptr<Botan::HashFunction> hashFunction;
	
	kj::UnwindDetector ud;
};

struct BlobReader {
	BlobReader(Blob& blob);
	
	bool read(kj::ArrayPtr<byte> output);
	inline size_t remainingOut() { return decompressor.remainingOut(); }
	
private:
	int64_t id;
	int64_t currentChunkNo = 0;
	
	Blob blob;
	
	Decompressor decompressor;
	sqlite::Statement readStatement;
};

struct ObjectDB : public kj::Refcounted {
	using capnp::Capability;
	using capnp::AnyPointer;
	
	//! Checks whether the given capability has been exported (but not yet finished processing)
	template<typename T>
	Maybe<T> findExport(T original);
	
	void exportObject(kj::StringPtr path, Capability::Client object);
	DataRef<Capability>::Client loadObject(kj::StringPtr path);
	
private:
	Maybe<Capability::Client> findExportInternal(Capability::Client cap);
	
	int64_t exportObject(Capability::Client cap);
	
	//! Clients that are currently in the process of being exported
	std::unordered_map<ClientHook*, int64_t> exports;
	
	//! These promises tell us when the object we have might be worth
	// looking into again.
	std::unordered_map<int64_t, ForkedPromise<void>> whenResolved;
	
	kj::TaskSet exportTasks;
};

//! Represents an object in the object database, as well as the permission to access it
struct DBObject : public kj::Refcounted {
	const int64_t id;
	
	// Manages the object inside the object database
	
};

struct DBExport : public DataRef<capnp::AnyPointer>::Server {
public:
	
private:
	//! Where to pass a reference to the import object 
	Own<PromiseFulfiller<Capability::Client>> forwardFulfiller;
	
	void fulfill(Capability::Client target);
	void reject(kj::Exception& exception);
	
	Own<ObjectDB> parent;
};

// ==================================== Inline implementation ===================================

BlobReader Blob::open() { return BlobReader(*this); }

template<typename T>
Maybe<T> ObjectDB::findExport(T original) {
	KJ_IF_MAYBE(pExp, findExportInternal(original)) {
		return pExp.as<capnp::FromClient<T>>();
	}
	
	return nullptr;
}