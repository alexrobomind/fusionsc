#pragma once

#include <unordered_map>
#include <botan/hash.h>
#include <capnp/capability.h>

#include <fsc/odb.capnp.h>

#include "db.h"
#include "compression.h"

namespace fsc { namespace odb {
	
struct BlobStore;
struct Blob;
struct BlobReader;
struct BlobBuilder;

struct ObjectDB;
struct DBObject;

struct BlobStore : public kj::Refcounted {
	using Statement = sqlite::Statement;
	
	Statement createBlob;
	Statement setBlobHash;
	Statement findBlob;
	Statement getBlobHash;
	
	Statement incRefcount;
	Statement decRefcount;
	
	Statement deleteIfOrphan;
	Statement createChunk;
	
	
	kj::String tablePrefix;
	Own<sqlite::Connection> conn;	
	const bool readOnly;

	inline Own<BlobStore> addRef() { return kj::addRef(*this); }
	
	Maybe<Blob> find(kj::ArrayPtr<const byte> hash);
	BlobBuilder create(size_t chunkSize);
		
private:
	BlobStore(sqlite::Connection& conn, kj::StringPtr tablePrefix, bool readOnly = false);
	
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
	
	void incRef();
	void decRef();
	
	kj::Array<const byte> hash();
	
	inline BlobReader open();
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
	bool read(kj::ArrayPtr<byte> output);
	inline size_t remainingOut() { return decompressor.remainingOut(); }
	
	BlobReader(Blob& blob);
	
private:	
	Blob blob;
	
	Decompressor decompressor;
	sqlite::Statement readStatement;
	sqlite::Statement::Query readQuery;
};

struct ObjectDB : public kj::Refcounted {
	using Capability = capnp::Capability;
	using AnyPointer = capnp::AnyPointer;
	using ClientHook = capnp::ClientHook;
	
	using Statement = sqlite::Statement;
	
	const kj::String filename;
	const kj::String tablePrefix;
	const bool readOnly;
	
	Statement createObject;
	Statement getInfo;
	Statement setInfo;
	Statement incRefcount;
	Statement decRefcount;
	Statement deleteObject;
	
	Statement insertRef;
	Statement listOutgoingRefs;
	Statement clearOutgoingRefs;
	
	Statement getRefcount;
	
	ObjectDB(kj::StringPtr filename, kj::StringPtr tablePrefix, bool readOnly = false);
	inline Own<ObjectDB> addRef() { return kj::addRef(*this); }
	
	//! Determines whether the given capability is outside the database, pointing to a DB object, or null
	OneOf<Capability::Client, Own<DBObject>, decltype(nullptr)> unwrap(Capability::Client cap);
	
	//! Wraps a DB object in a capability exposing its functionality.
	Object::Client wrap(Maybe<Own<DBObject>> obj);
	
	//! Checks the reference count of an object and deletes it is 0.
	void deleteIfOrphan(int64_t id);
	
	Folder::Client getRoot();
	
private:
	//! Replaces a DataRef with a variant pointing into the database
	// Note: This method is private because the database does not support dangling objects. Therefore, any creation of such pointers
	// must be put in a transaction with their integration into the file hierarchy. 
	Object::Client download(DataRef<AnyPointer>::Client object);
	
	//! Creates a new slot for a DataRef and initiates a download task
	Own<DBObject> startDownloadTask(DataRef<AnyPointer>::Client object);
	
	//! Performs the download operations required to store the DataRef
	Promise<void> downloadTask(DataRef<AnyPointer>::Client src, DBObject& dst);
	
	//! Clients that are currently in the process of being exported
	std::unordered_map<ClientHook*, int64_t> activeDownloads;
	
	//! These promises tell us when the object we have might be worth
	// looking into again.
	std::unordered_map<int64_t, ForkedPromise<void>> whenResolved;
	
	//! Creates a new connection to the same database
	Own<sqlite::Connection> forkConnection(bool readOnly = true);
	
	void createRoot();
		
	kj::TaskSet downloadTasks;
	
	Own<BlobStore> blobStore;
	Own<sqlite::Connection> conn;
	
	bool shared;
	
	struct TransmissionProcess;
	struct TransmissionReceiver;
	
	struct ObjectImpl;
	struct ObjectHook;
	
	friend class DBObject;
};

//! Represents an object in the object database, as well as the permission to access it
struct DBObject : public kj::Refcounted {
private:
	struct CreationToken {};

public:
	DBObject(ObjectDB& parent, int64_t id, const CreationToken&);
	~DBObject();
	
	void load();
	void save();
	
	ObjectInfo::Builder info;
	
	Promise<void> whenUpdated();
	
	inline ObjectDB& getParent() { return *parent; }
	inline Own<DBObject> addRef() { return kj::addRef(*this); }
	
private:
	const int64_t id;
	Own<ObjectDB> parent;
	
	Own<capnp::MallocMessageBuilder> infoHolder;
	
	friend class ObjectDB;
};

Folder::Client openObjectDB(kj::StringPtr folder);

// ==================================== Inline implementation ===================================

BlobReader Blob::open() { return BlobReader(*this); }

}}