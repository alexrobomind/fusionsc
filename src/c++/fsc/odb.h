#pragma once

#include <unordered_map>
#include <botan/hash.h>
#include <capnp/capability.h>

#include <fsc/odb.capnp.h>

#include "db.h"
#include "compression.h"

namespace fsc { namespace odb {
	
struct ObjectDB;
struct DBObject;

struct ObjectDB : public kj::Refcounted {
	using Capability = capnp::Capability;
	using AnyPointer = capnp::AnyPointer;
	using ClientHook = capnp::ClientHook;
	
	using Statement = db::PreparedStatement;
	
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
	
	ObjectDB(kj::StringPtr filename, kj::StringPtr tablePrefix = "objectDB", bool readOnly = false);
	inline ~ObjectDB() {}
	
	inline Own<ObjectDB> addRef() { return kj::addRef(*this); }
	
	//! Determines whether the given capability is outside the database, pointing to a DB object, or null
	OneOf<Capability::Client, Own<DBObject>, decltype(nullptr)> unwrap(Capability::Client cap);
	
	//! Wraps a DB object in a capability exposing its functionality.
	Object::Client wrap(Maybe<Own<DBObject>> obj);
	
	//! Checks the reference count of an object and deletes it is 0.
	void deleteIfOrphan(int64_t id);
	
	Folder::Client getRoot();
	
	inline void cancelDownloads() { canceler.cancel("Downloads canceled"); whenResolved.clear(); }
	Promise<void> drain();
	
private:
	//! Replaces a DataRef with a variant pointing into the database
	// Note: This method is private because the database does not support dangling objects. Therefore, any creation of such pointers
	// must be put in a transaction with their integration into the file hierarchy. 
	Object::Client download(DataRef<AnyPointer>::Client object);
	
	//! Creates a new slot for a DataRef and initiates a download task
	Own<DBObject> startDownloadTask(DataRef<AnyPointer>::Client object);
	
	//! Performs the download operations required to store the DataRef
	Promise<void> downloadTask(DataRef<AnyPointer>::Client src, int64_t id);
	
	//! These promises tell us when the object we have might be worth
	// looking into again.
	std::unordered_map<int64_t, ForkedPromise<void>> whenResolved;
	
	//! Creates a new connection to the same database
	Own<sqlite::Connection> forkConnection(bool readOnly = true);
	
	Own<DBObject> open(int64_t id);
	
	void createRoot();
	
	Own<BlobStore> blobStore;
	Own<sqlite::Connection> conn;
	
	bool shared;
	
	kj::Canceler canceler;
	
	struct TransmissionReceiver;
	struct DownloadProcess;
	
	struct ObjectImpl;
	struct ObjectHook;
	
	friend class DBObject;
};

struct TransmissionProcess;

//! Represents an object in the object database, as well as the permission to access it
struct DBObject : public kj::Refcounted {
public:
	DBObject(ObjectDB& parent, int64_t id);
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

// ==================================== Inline implementation ===================================

Own<BlobReader> Blob::open() { return kj::heap<BlobReader>(*this); }

}}