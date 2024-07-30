#include "odb.h"
#include "data.h"
#include "blob-store.h"
#include "thread-pool.h"

#include <capnp/rpc.capnp.h>
#include <fsc/warehouse-internal.capnp.h>

#include <capnp/serialize-packed.h>

#include <kj/map.h>

#include <list>
#include <set>
#include <typeinfo>

using kj::str;

using namespace capnp;

using ::fsc::internal::ObjectInfo;

namespace fsc { namespace {
	
template<typename T>
auto withODBBackoff(T func) {
	return kj::evalLater([func = mv(func)]() mutable {
		return withBackoff(10 * kj::MILLISECONDS, 5 * kj::MINUTES, 2, mv(func));
	});
}

kj::Exception fromProto(rpc::Exception::Reader proto) { 
	kj::Exception::Type type;
	switch(proto.getType()) {
		#define HANDLE_VAL(val) \
			case rpc::Exception::Type::val: \
				type = kj::Exception::Type::val; \
				break;
				
		HANDLE_VAL(FAILED)
		HANDLE_VAL(OVERLOADED)
		HANDLE_VAL(DISCONNECTED)
		HANDLE_VAL(UNIMPLEMENTED)
		
		#undef HANDLE_VAL
	}
	
	kj::Exception result(type, "remote", -1, str(proto.getReason()));
	
	if(proto.hasTrace()) {
		result.setRemoteTrace(str(proto.getTrace()));
	}
	
	return result;
}

Temporary<rpc::Exception> toProto(const kj::Exception& e) {
	Temporary<rpc::Exception> result;
	
	switch(e.getType()) {
		#define HANDLE_VAL(val) \
			case kj::Exception::Type::val: \
				result.setType(rpc::Exception::Type::val); \
				break;
				
		HANDLE_VAL(FAILED)
		HANDLE_VAL(OVERLOADED)
		HANDLE_VAL(DISCONNECTED)
		HANDLE_VAL(UNIMPLEMENTED)
		
		#undef HANDLE_VAL
	}
	
	result.setReason(e.getDescription());
	
	if(e.getRemoteTrace() != nullptr) {
		result.setTrace(e.getRemoteTrace());
	}
	
	return result;
}

// ============================= SQL Database Structure =================================

struct ObjectDBBase {
	using Statement = db::PreparedStatement;
	
	ObjectDBBase(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly);
	
	Statement createObject;
	Statement setInfo;
	Statement incRefcount;
	Statement decRefcount;
	Statement deleteObject;
	Statement restoreObject;
	
	Statement insertRef;
	Statement clearOutgoingRefs;
	
	Statement updateFolderEntry;
	Statement deleteFolderEntry;
	
	Statement getInfo;
	Statement listOutgoingRefs;
	Statement listFolderEntries;
	Statement getFolderEntry;
	Statement getRefcount;
	
	Statement getBlob;
	Statement setBlob;
	
	Statement readNewestId;
	
	Statement createRoot;
	Statement findRoot;
	
	kj::String tablePrefix;
	Own<db::Connection> conn;
	const bool readOnly;
	Own<BlobStore> blobStore;
	
	Maybe<int64_t> getNewestId();
};

ObjectDBBase::ObjectDBBase(db::Connection& paramConn, kj::StringPtr paramTablePrefix, bool paramReadOnly) :
	tablePrefix(kj::heapString(paramTablePrefix)),
	conn(paramConn.addRef()),
	readOnly(paramReadOnly),
	blobStore(createBlobStore(*conn, tablePrefix, readOnly))
{		
	auto objectsTable = str(tablePrefix, "_objects");
	auto refsTable = str(tablePrefix, "_object_refs");
	auto folderEntriesTable = str(tablePrefix, "_folder_entries");
	auto blobsTable = str(tablePrefix, "_blobs");
	auto rootsTable = str(tablePrefix, "_roots");
	
	if(!readOnly) {
		// The default message is 0x'00 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00' (1 segment with only a null root)
		// In packed encoding, this is 0x'10 01 00 00'
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", objectsTable, " ("
			  "id INTEGER PRIMARY KEY AUTOINCREMENT,"
			  "info BLOB DEFAULT x'10010000',"
			  "refcount INTEGER DEFAULT 0,"
			  "blobId INTEGER DEFAULT NULL REFERENCES ", blobsTable, "(id) ON DELETE SET NULL ON UPDATE CASCADE"
			")"
		));
		conn -> exec(str(
			"CREATE INDEX IF NOT EXISTS ", tablePrefix, "_objects_by_blob ON ", objectsTable, "(blobId)"
		));
		
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", refsTable, " ("
			  "parent INTEGER REFERENCES ", objectsTable, "(id) ON DELETE CASCADE ON UPDATE CASCADE,"
			  "slot INTEGER,"
			  "child INTEGER REFERENCES ", objectsTable, "(id) ON UPDATE CASCADE"
			")"
		));
		conn -> exec(str(
			"CREATE UNIQUE INDEX IF NOT EXISTS ", tablePrefix, "_index_refs_by_parent ON ", refsTable, "(parent, slot)"
		));
		conn -> exec(str(
			"CREATE INDEX IF NOT EXISTS ", tablePrefix, "_index_refs_by_child ON ", refsTable, "(child)"
		));
		
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", folderEntriesTable, " ("
			  "parent INTEGER REFERENCES ", objectsTable, "(id) ON DELETE CASCADE ON UPDATE CASCADE,"
			  "name TEXT,"
			  "child INTEGER REFERENCES ", objectsTable, "(id) ON UPDATE CASCADE"
			")"
		));
		conn -> exec(str(
			"CREATE UNIQUE INDEX IF NOT EXISTS ", tablePrefix, "_index_folder_entries_by_parent ON ", folderEntriesTable, "(parent, name)"
		));
		conn -> exec(str(
			"CREATE INDEX IF NOT EXISTS ", tablePrefix, "_index_folder_entries_by_child ON ", folderEntriesTable, "(child)"
		));
		
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", rootsTable, " ("
			  "id INTEGER REFERENCES ", objectsTable, "(id),"
			  "name TEXT"
			")"
		));
		conn -> exec(str(
			"CREATE UNIQUE INDEX IF NOT EXISTS ", tablePrefix, "_index_roots_by_name ON ", rootsTable, "(name)"
		));
		conn -> exec(str(
			"CREATE INDEX IF NOT EXISTS ", tablePrefix, "_index_roots_by_id ON ", rootsTable, "(id)"
		));
		
		
		createObject = conn -> prepare(str("INSERT INTO ", objectsTable, " DEFAULT VALUES"));
		setInfo = conn -> prepare(str("UPDATE ", objectsTable, " SET info = ?2 WHERE id = ?1"));
		incRefcount = conn -> prepare(str("UPDATE ", objectsTable, " SET refcount = refcount + 1 WHERE id = ?"));
		decRefcount = conn -> prepare(str("UPDATE ", objectsTable, " SET refcount = refcount - 1 WHERE id = ?"));
		deleteObject = conn -> prepare(str("DELETE FROM ", objectsTable, " WHERE id = ?"));
		restoreObject = conn -> prepare(str("INSERT INTO ", objectsTable, " (id) VALUES (?)"));
		
		setBlob = conn -> prepare(str("UPDATE ", objectsTable, " SET blobId = ?2 WHERE id = ?1"));
		
		insertRef = conn -> prepare(str("INSERT INTO ", refsTable, " (parent, slot, child) VALUES (?, ?, ?)"));
		clearOutgoingRefs = conn -> prepare(str("DELETE FROM ", refsTable, " WHERE parent = ?"));
		
		updateFolderEntry = conn -> prepare(str("INSERT OR REPLACE INTO ", folderEntriesTable, " (parent, name, child) VALUES (?, ?, ?)"));
		deleteFolderEntry = conn -> prepare(str("DELETE FROM ", folderEntriesTable, " WHERE parent = ? AND name = ?"));
		
		createRoot = conn -> prepare(str("INSERT INTO ", rootsTable, " (id, name) VALUES (?, ?)"));
	}
	
	getInfo = conn -> prepare(str("SELECT info FROM ", objectsTable, " WHERE id = ?"));
	getRefcount = conn -> prepare(str("SELECT refcount FROM ", objectsTable, " WHERE id = ?"));
	getBlob = conn -> prepare(str("SELECT blobId FROM ", objectsTable, " WHERE id = ?"));
	findRoot = conn -> prepare(str("SELECT id FROM ", rootsTable, " WHERE name = ?"));
	
	listOutgoingRefs = conn -> prepare(str("SELECT child FROM ", refsTable, " WHERE parent = ? ORDER BY slot"));
	listFolderEntries = conn -> prepare(str("SELECT name, child FROM ", folderEntriesTable, " WHERE parent = ? ORDER BY name"));
	getFolderEntry =  conn -> prepare(str("SELECT child FROM ", folderEntriesTable, " WHERE parent = ? AND name = ?"));
	
	readNewestId = conn -> prepare(str("SELECT seq FROM sqlite_sequence WHERE name = ?"));
}

Maybe<int64_t> ObjectDBBase::getNewestId() {
	auto q = readNewestId.bind(kj::str(tablePrefix, "_objects").asPtr());
	if(!q.step())
		return nullptr;
	
	return q[0].asInt64();
}

// ============================== Database access ===================================

struct ObjectDB;
struct ObjectDBSnapshot;
struct ObjectDBEntry;
struct ObjectDBHook;

/** Handle to an object expected to be present in the database.
 *
 * This class serves as the central access point for the retrieval and modification
 * of objects inside the database. Handles to the object are used to back capabilities
 * and to track the structure of the database.
 *
 * Besides serving as access points to open the current version of the object for
 * reading and writing, every entry handle also maintains a "most recently seen" view
 * of the object pointing to a database snapshot where it is known to have existed.
 * This is crucial to guarantee access to objects that might have been intermediately
 * deleted by the current or other process without expensive database-side tracking
 * of open references.
 *
 * This "preserved" view can be synchronized to the most recent frozen view of the
 * backing database - provided that the object exists in that view. Note that objects
 * can be missing from this view because they have been deleted or created.
 *
 * Entry handles maintain a "liveness" tracker to avoid frequent synchronization of
 * objects that were deleted in the database (and thus can receive no more updates).
 */ 
struct ObjectDBEntry : public kj::Refcounted {
	ObjectDBEntry(ObjectDB& parent, int64_t id, Maybe<Own<ObjectDBSnapshot>> snapshot);
	~ObjectDBEntry();
	
	Own<ObjectDBEntry> addRef() { return kj::addRef(*this); }
	
	//! Update the preserved snapshot to the most recent snapshot (if possible)
	void trySync(bool force = false);
	
	/**
	  * Register a promise that fires when the preserved view moves to a new snapshot
	  * and that throws when the object gets deleted before a new snapshot sees it.
	  */
	Promise<void> whenUpdated();
	
	//! Load the last recently seen representation (read-only) of the object.
	Own<ObjectInfo::Reader> loadPreserved(bool shallow = false);
	
	//! Read the preserved snapshot
	ObjectDBSnapshot& getSnapshot();
	
	/** Check whether the object has a "last recently seen" version.
	 *
	 * This function should always return true in interfaces created by
	 * ObjectDBHook
	 */
	bool hasPreserved() { return snapshot != nullptr; }
	
	struct Accessor;
	
	/** Open a read-modify-write view of the object
	 *
	 * Note: The returned objects maintains an open transaction
	 */
	Own<Accessor> open();
	
	ObjectDB& getDb() { return *parent; }
	
	//! Checks whether the object exists in the live DB
	bool exists();
	
	//! Restores a non-existing object into the live DB
	void restore();

public:
	const int64_t id;

private:
	Own<ObjectInfo::Reader> doLoad(Maybe<ObjectDBSnapshot&> snapshot, bool shallow);
	
	bool isLive() { return liveLink.isLinked(); }
	
	Own<ObjectDB> parent;
	Maybe<Own<ObjectDBSnapshot>> snapshot;
		
	Maybe<Own<PromiseFulfiller<void>>> updateFulfiller;
	ForkedPromise<void> updatePromise;
	
	kj::ListLink<ObjectDBEntry> liveLink;
	friend class ObjectDB;
	friend class ObjectEntryData;
};

/** Read-modify-write accessor for database objects
 *
 * This class maintains a builder that can be modified to change the state of the object.
 * Upon destruction (or release(true) ), the modified representation will be written back
 * into the database. 
 */
struct ObjectDBEntry::Accessor : public ObjectInfo::Builder {
	Accessor(ObjectDBEntry& e);
	~Accessor();
	
	/**
	 * Release the accessor, optionally with (true) or without (false) writing the
	 * object representation back to the database.
	 */
	void release(bool doSave) { if(doSave && !released) { save(); released = true; } }
	
private:
	void load();
	void save();
	
	kj::UnwindDetector ud;
	
	Own<ObjectDBEntry> target;
	
	MallocMessageBuilder messageBuilder;
	BuilderCapabilityTable capTable;
	
	bool released = false;
};

/** Object database
 *
 * This class maintains the main connection to the database, as well as a frozen
 * snapshot view of the database, that can be repeatedly updated.
 */
struct ObjectDB : public ObjectDBBase, kj::Refcounted, Warehouse::Server {
	ObjectDB(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly);
	~ObjectDB();
	
	Own<ObjectDB> addRef() { return kj::addRef(*this); }
	
	//! Returns a current frozen representation of the database
	ObjectDBSnapshot& getCurrentSnapshot();
	
	/**
	 * Invalidates the current snapshot, causing getCurrentSnapshot() to return a new snapshot
	 * on next call
	 */
	void changed();
	
	/**
	 * Synchronize (ObjectDBEntry::trySync) all currently alive entry handles to the
	 * current snapshot.
	 */
	void syncAll();
	
	/** Open entry handle.
	 *
	 * Open the object with the given ID, optionally providing a snapshot where the object is
	 * expected to be alive. If this is true, this ensures that this object will always have
	 * a valid preserved representation.
	 */
	Own<ObjectDBEntry> open(int64_t id, Maybe<ObjectDBSnapshot&> snapshot = nullptr);
	
	/** Unwrap capability client
	 *
	 * This method analyzes the client hook chain of the given capapability, and checks whether
	 * it is
	 *  - Pointing to a (resolved or unresolved) object in this database
	 *  - Is a "null" object
	 *  - Is none of the above
	 *
	 * Depending on the result, an appropriate object is returned.
	 */
	OneOf<Own<ObjectDBEntry>, Capability::Client, decltype(nullptr)> unwrap(Capability::Client);
	
	//! Wraps a DB object (or null value) as a capability client.
	Capability::Client wrap(Maybe<Own<ObjectDBEntry>> e);
	
	//! Unwraps an object KNOWN to be pointing to a database entry
	Own<ObjectDBEntry> unwrapValid(Capability::Client c) { return unwrap(c).get<Own<ObjectDBEntry>>(); }
	
	//! Writes presently known information about the object into the target entry
	void exportStoredObject(Capability::Client c, Warehouse::StoredObject::Builder);
	
	//! Checks whether the object's reference count is 0 If yes, deletes it.
	void deleteIfOrphan(int64_t id);
	
	/** Creates a new object in the database.
	 *
	 * Inserts a new entry with refcount 0 into the DB table.
	 *
	 * \warning Make sure that you call this in a transaction  with the functions actually
	 *          storing the reference. Otherwise, the object might get deleted at any
	 *          time, corrupting the database.
	 */
	Own<ObjectDBEntry> create();
	
	//! Creates the root entry if not present
	Promise<void> getRoot(GetRootContext ctx) override;
	
	void writeLock();
	Promise<void> writeBarrier();
	
	Own<db::Transaction> writeTransaction();
	
	/*template<typename F>
	kj::PromiseForResult<F, void> runWriteTask(F&& f);*/

private:
	// std::list<kj::Function<void()>> writeTasks;
	
	//! Task that periodically cycles the read snapshots
	kj::Promise<void> syncTask();
	
	//! Task that periodically runs write transactions
	// kj::Promise<void> writer();
	
	Maybe<Own<ObjectDBSnapshot>> currentSnapshot;
	
	kj::List<ObjectDBEntry, &ObjectDBEntry::liveLink> liveEntries;
	friend class ObjectDBEntry;
	
	kj::TaskSet importTasks;
	friend class ImportContext;
	
	kj::Promise<void> syncPromise;
	// kj::Promise<void> writePromise;
	
	struct WriteLock {
		ObjectDB& parent;
		
		inline WriteLock(ObjectDB& db) : parent(db) {
			parent.conn -> exec("BEGIN IMMEDIATE");
		}
		
		inline ~WriteLock() {
			parent.conn -> exec("COMMIT");
		}
	};
	
	Maybe<WriteLock> hasWriteLock = nullptr;
	std::list<Own<kj::PromiseFulfiller<void>>> lockWaiters;
	
	uint32_t checkpointCounter = 0;
};

//! Snapshot to a frozen representation of the database
struct ObjectDBSnapshot : public ObjectDBBase, kj::Refcounted {
	ObjectDBSnapshot(ObjectDB& base);
	
	Own<ObjectDBSnapshot> addRef() { return kj::addRef(*this); }
	
	db::Savepoint savepoint;
};


/** Client hook to create capabilities in database objects
 *
 * We use a raw client hook instead of the ServerSet mechanism here, because server
 * sets do not have the ability to check the "already resolved" portion of the hooks
 * without invocation of the async machinery.
 *
 * This class serves both as a marker for database-backed capabilities and as the driver
 * for initial resolution, distunguishing unresolved objects, exceptions, and resolved
 * objects, as well as performing dispatch to different capability servers depending
 * on the resolved object type.
 */
struct ObjectDBHook : public ClientHook, kj::Refcounted {	
	// Interface methods
	// These methods all defer to an inner client created from a resolve task
	// promise.
	Request<AnyPointer, AnyPointer> newCall(
		uint64_t interfaceId,
		uint16_t methodId,
		kj::Maybe<MessageSize> sizeHint,
		CallHints hints
	) override {
		return inner -> newCall(interfaceId, methodId, mv(sizeHint), mv(hints));
	}

	VoidPromiseAndPipeline call(
		uint64_t interfaceId,
		uint16_t methodId,
		kj::Own<CallContextHook>&& context,
		CallHints hints
	) override {
		return inner -> call(interfaceId, methodId, mv(context), mv(hints));
	}

	kj::Maybe<ClientHook&> getResolved() override {	return nullptr; }
	kj::Maybe<kj::Promise<kj::Own<ClientHook>>> whenMoreResolved() override {
		if(resolved.is<Resolved>())
			return nullptr;
		
		if(resolved.is<kj::Exception>())
			return kj::Promise<kj::Own<ClientHook>>(cp(resolved.get<kj::Exception>()));
		
		return inner -> whenResolved()
		.then([ref = addRef()]() mutable {
			return mv(ref);
		});
	}

	virtual kj::Own<ClientHook> addRef() override { return kj::addRef(*this); }
	const void* getBrand() override { return &BRAND; }
	kj::Maybe<int> getFd() override { return nullptr; }
	
	// Implementation method
	ObjectDBHook(Own<ObjectDBEntry> objectIn);
	~ObjectDBHook();
	
	//! Checks whether the backing object has resolved / thrown.
	OneOf<decltype(nullptr), Capability::Client, kj::Exception> checkResolved();
	
	//! Task to periodically check the backing object and resolve eventually.
	Promise<Own<ClientHook>> resolveTask();

	// Member
	inline static const uint BRAND = 0;
	Own<ObjectDBEntry> entry;
	Own<ClientHook> inner;
	
	struct Resolved {};
	struct Unresolved {};
	OneOf<Unresolved, Resolved, kj::Exception> resolved = Unresolved();
};

struct ImportErrorHandler : public kj::TaskSet::ErrorHandler {
	void taskFailed(kj::Exception&& exception) override {
		KJ_LOG(WARNING, "Import task failed", exception);
	}
	
	static ImportErrorHandler INSTANCE;
};

ImportErrorHandler ImportErrorHandler::INSTANCE;

//! Glue function that creates the access interface for entries.
OneOf<decltype(nullptr), Capability::Client, kj::Exception> createInterface(ObjectDBEntry& e, kj::Badge<ObjectDBHook> badge);

// class ObjectDBBase

// class ObjectDB

ObjectDB::ObjectDB(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly) :
	ObjectDBBase(conn, tablePrefix, readOnly),
	importTasks(ImportErrorHandler::INSTANCE),
	syncPromise(kj::evalLater([this]() { return syncTask(); }).eagerlyEvaluate(nullptr))
	// writePromise(READY_NOW)
{}

ObjectDB::~ObjectDB() {
}

Promise<void> ObjectDB::syncTask() {
	changed();
	// syncAll();
	
	return getActiveThread().timer().afterDelay(1 * kj::SECONDS)
	.then([this]() { return syncTask(); });
}

Promise<void> ObjectDB::getRoot(GetRootContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		
		// Look for root
		{
			auto& snapshot = getCurrentSnapshot();
			auto q = snapshot.findRoot.bind(ctx.getParams().getName());
			if(q.step()) {
				ctx.initResults().setRoot(wrap(open(q[0].asInt64(), snapshot)).castAs<Warehouse::Folder>());
				return;
			}
		}
		
		KJ_REQUIRE(!readOnly, "The requested root does not exist and the database is read-only");
		
		auto t = writeTransaction();
		
		// Look again
		auto q = findRoot.bind(ctx.getParams().getName());
		if(q.step()) {
			ctx.initResults().setRoot(wrap(open(q[0].asInt64())).castAs<Warehouse::Folder>());
			return;
		}
		
		// Still not found? Create new.
		auto newRoot = create();
		auto acc = newRoot -> open();
		acc -> setFolder();
		acc -> release(true);
		
		createRoot.insert(newRoot -> id, ctx.getParams().getName());
		incRefcount(newRoot -> id);
		
		ctx.initResults().setRoot(wrap(mv(newRoot)).castAs<Warehouse::Folder>());
	});
}
	
ObjectDBSnapshot& ObjectDB::getCurrentSnapshot() {
	KJ_IF_MAYBE(p, currentSnapshot) {
		return **p;
	}
	
	// We need to open a fresh snapshot
	auto newSnapshot = kj::refcounted<ObjectDBSnapshot>(*this);
	currentSnapshot = newSnapshot -> addRef();
	
	return *newSnapshot;
}

void ObjectDB::changed() {
	if(checkpointCounter++ >= 300) {
		try {
			syncAll();
			
			// Ideally, now all readers point to the "present state"
			// of the database at the end of the WAL
			
			conn -> exec("PRAGMA wal_checkpoint(PASSIVE)");
			
			// Now, the database file is up do date
			
			syncAll();
			
			// Now, all readers point to the database file and the WAL
			// is no longer in use.
			// This allows us to restart the WAL
			conn -> exec("PRAGMA wal_checkpoint(RESTART)");
			
			checkpointCounter = 0;
		} catch(kj::Exception& e) {
			KJ_LOG(WARNING, "Error during checkpoint", e);
		}
	} else {
		syncAll();
	}
}

void ObjectDB::syncAll() {
	currentSnapshot = nullptr;
	for(auto& e : liveEntries) {
		try {
			e.trySync();
		} catch(kj::Exception e) {
		}
	}
}

Own<ObjectDBEntry> ObjectDB::open(int64_t id, Maybe<ObjectDBSnapshot&> snapshot) {
	KJ_IF_MAYBE(pSnapshot, snapshot) {
		return kj::refcounted<ObjectDBEntry>(*this, id, pSnapshot -> addRef());
	}
	return kj::refcounted<ObjectDBEntry>(*this, id, nullptr);
}

Capability::Client ObjectDB::wrap(Maybe<Own<ObjectDBEntry>> e) {
	KJ_IF_MAYBE(pEntry, e) {
		return Capability::Client(kj::refcounted<ObjectDBHook>(mv(*pEntry)));
	}
	return nullptr;
}

OneOf<Own<ObjectDBEntry>, Capability::Client, decltype(nullptr)> ObjectDB::unwrap(Capability::Client clt) {
	// Resolve to innermost hook
	// If any of the hooks encountered is an object hook, use that
	Own<capnp::ClientHook> hook = capnp::ClientHook::from(kj::cp(clt));
	
	ClientHook* inner = hook.get();
	
	// Find INNERmost hook that matches
	Own<ObjectDBHook> matchingHook;
	while(true) {
		if(inner -> getBrand() == &ObjectDBHook::BRAND) {
			auto asObjectHook = static_cast<ObjectDBHook*>(inner);
			
			if(asObjectHook -> entry -> parent.get() == this) {
				matchingHook = kj::addRef(*asObjectHook);
			}
		}
		
		KJ_IF_MAYBE(pHook, inner -> getResolved()) {
			inner = pHook;
		} else {
			break;
		}
	}
	
	if(matchingHook.get() != nullptr) {
		return matchingHook -> entry -> addRef();
	} /*else if (inner -> whenMoreResolved() != nullptr) {
		KJ_DBG("unwrap() encountered unresolved promise outside DB");
	} */
	
	// ... otherwise check whether it is a null cap ... 
	if(inner -> isNull()) {
		return nullptr;
	}
	
	// ... turns out we can't unwrap this.
	return mv(clt);
}

void ObjectDB::exportStoredObject(Capability::Client c, Warehouse::StoredObject::Builder out) {
	auto entry = unwrapValid(c);
	entry -> trySync(true);
	
	out.setAsGeneric(c.castAs<Warehouse::GenericObject>());
	
	if(!entry -> hasPreserved()) {
		if(entry -> isLive()) {
			out.setUnresolved();
		} else {
			out.setDead();
		}
		
		return;
	}
	
	auto data = entry -> loadPreserved();
	switch(data -> which()) {
		case ObjectInfo::UNRESOLVED:
			if(entry -> isLive()) {
				out.setUnresolved();
			} else {
				out.setDead();
			}
		
			return;
			
		case ObjectInfo::NULL_VALUE:
			out.setNullValue();
			return;
			
		case ObjectInfo::EXCEPTION:
			out.setException(data -> getException());
			return;
			
		case ObjectInfo::LINK:
			exportStoredObject(data -> getLink(), out);
			return;
			
		case ObjectInfo::DATA_REF: {
			auto info = out.initDataRef();
			if(data -> getDataRef().getDownloadStatus().isFinished())
				info.getDownloadStatus().setFinished();
			info.setAsRef(c.castAs<DataRef<>>());
			
			return;
		}
		
		case ObjectInfo::FOLDER:
			out.setFolder(c.castAs<Warehouse::Folder>());
			return;
		
		case ObjectInfo::FILE:
			out.setFile(c.castAs<Warehouse::File<>>());
			return;
	}
	
	KJ_FAIL_REQUIRE("Unknown object type in database");
}

void ObjectDB::deleteIfOrphan(int64_t id) {
	auto t = writeTransaction();
	
	// Make sure that the object is not in use
	auto rc = getRefcount.bind(id);
	KJ_REQUIRE(rc.step(), "Internal error, refcount not found");
	
	if(rc[0].asInt64() > 0)
		return;
	
	Maybe<int64_t> blobId;
	auto gbi = getBlob.bind(id);
	gbi.step();
	
	if(!gbi[0].isNull()) {
		blobId = gbi[0].asInt64();
	}
		
	// Scan outgoing refs
	std::set<int64_t> idsToCheck;
	{
		auto outgoingRefs = listOutgoingRefs.bind(id);
		while(outgoingRefs.step()) {
			// Skip NULL
			if(outgoingRefs[0].isNull()) {
				continue;
			}
			
			auto target = outgoingRefs[0].asInt64();
			decRefcount(target);
			idsToCheck.insert(target);
		}
	}
	
	deleteObject(id);
	
	KJ_IF_MAYBE(pBlobId, blobId) {
		blobStore -> get(*pBlobId) -> decRef();
	}
	
	// Check if we have capabilities without refs
	for(int64_t id : idsToCheck) {
		deleteIfOrphan(id);
	}	
}

Own<ObjectDBEntry> ObjectDB::create() {
	KJ_REQUIRE(conn -> inTransaction(), "INTERNAL ERROR - ObjectDB::create() must be called inside a transaction");
	
	int64_t id = createObject.insert();
	auto result = open(id, nullptr);
	
	return result;
}

Own<ObjectDBEntry::Accessor> ObjectDBEntry::open() {
	return kj::heap<Accessor>(*this);
}

void ObjectDB::writeLock() {
	KJ_REQUIRE(!readOnly);
	
	// Increase checkpoint counter every time a task
	// tries to acquire the write lock
	++checkpointCounter;
	
	if(hasWriteLock != nullptr)
		return;
	
	hasWriteLock.emplace(*this);
	
	importTasks.add(kj::evalLast([this]() {
		KJ_IF_MAYBE(pErr, kj::runCatchingExceptions([this]() {
			hasWriteLock = nullptr;
		})) {
			KJ_LOG(WARNING, "Write op failed", *pErr);
			
			for(auto& e : lockWaiters)
				e -> reject(kj::cp(*pErr));
		} else {
			for(auto& e : lockWaiters)
				e -> fulfill();
		}
		
		// Make sure we really are clear (double fail should not happen)
		hasWriteLock = nullptr;
		lockWaiters.clear();
		
		changed();
	}));
}

Promise<void> ObjectDB::writeBarrier() {
	KJ_IF_MAYBE(pWl, hasWriteLock) {
		auto paf = kj::newPromiseAndFulfiller<void>();
		lockWaiters.push_back(mv(paf.fulfiller));
		return mv(paf.promise);
	}
	
	return READY_NOW;
}

Own<db::Transaction> ObjectDB::writeTransaction() {
	writeLock();
	return kj::heap<db::Transaction>(*conn);
}

// class ObjectDBSnapshot

ObjectDBSnapshot::ObjectDBSnapshot(ObjectDB& base) :
	ObjectDBBase(*base.conn -> fork(true), base.tablePrefix, true),
	savepoint(*conn)
{
}

// class ObjectDBEntry

ObjectDBEntry::ObjectDBEntry(ObjectDB& parent, int64_t id, Maybe<Own<ObjectDBSnapshot>> paramSnapshot) :
	parent(parent.addRef()), id(id), updatePromise(nullptr)
{
	KJ_IF_MAYBE(pSnap, paramSnapshot) {
		snapshot = (**pSnap).addRef();
	}
	
	parent.liveEntries.add(*this);
}

ObjectDBEntry::~ObjectDBEntry() {
	if(isLive()) {
		parent -> liveEntries.remove(*this);
	}
}

void ObjectDBEntry::trySync(bool force) {
	if(!isLive() && !force)
		return;
	
	ObjectDBSnapshot& dbSnapshot = parent -> getCurrentSnapshot();
	
	KJ_IF_MAYBE(pMySnap, snapshot) {
		// We are already on the current snapshot
		if(pMySnap -> get() == &dbSnapshot)
			return;
	}
	
	// Check if the object exists IN THE SNAPSHOT
	auto& getRefcount = dbSnapshot.getRefcount;
	auto q = getRefcount.bind(id);
	
	if(!q.step()) {
		// Either object is dead, or not yet in the database
		KJ_IF_MAYBE(pId, dbSnapshot.getNewestId()) {
			if(*pId >= id) {
				// Object is dead
				KJ_IF_MAYBE(ppFulfiller, updateFulfiller) {
					(**ppFulfiller).reject(KJ_EXCEPTION(DISCONNECTED, "Object was deleted from database"));
				}
				
				if(isLive())
					parent -> liveEntries.remove(*this);
			}
		}
		
		return;
	}
	
	if(!isLive())
		parent -> liveEntries.add(*this);
	
	// Update to new snapshot
	snapshot = dbSnapshot.addRef();
	
	// Notify waiting updates
	KJ_IF_MAYBE(ppFulfiller, updateFulfiller) {
		(**ppFulfiller).fulfill();
		
		auto paf = kj::newPromiseAndFulfiller<void>();
		updateFulfiller = mv(paf.fulfiller);
		updatePromise = paf.promise.fork();
	}
}

ObjectDBSnapshot& ObjectDBEntry::getSnapshot() {
	KJ_IF_MAYBE(pMySnap, snapshot) {
		return **pMySnap;
	}
	
	KJ_FAIL_REQUIRE("Snapshot requested on object without snapshot");
}

Promise<void> ObjectDBEntry::whenUpdated() {
	if(!isLive()) {
		return KJ_EXCEPTION(DISCONNECTED, "Object was deleted from database");
	}
	
	if(updateFulfiller == nullptr) {
		auto paf = kj::newPromiseAndFulfiller<void>();
		updateFulfiller = mv(paf.fulfiller);
		updatePromise = paf.promise.fork();
	}
	
	return updatePromise.addBranch();
}

Own<ObjectInfo::Reader> ObjectDBEntry::doLoad(Maybe<ObjectDBSnapshot&> snapshotToUse, bool shallow) {
	auto getTargetBase = [&]() -> ObjectDBBase& {
		KJ_IF_MAYBE(pSnap, snapshotToUse) {
			return *pSnap;
		}
		
		return *parent;
	};
	ObjectDBBase& targetBase = getTargetBase();
	
	// Start transaction
	// db::Transaction t(*targetBase.conn);
	KJ_REQUIRE(targetBase.conn -> inTransaction(), "Internal error: Accessor loaded outside transaction");
	
	auto q = targetBase.getInfo.bind(id);
	KJ_REQUIRE(q.step(), "Object not present in database");
	
	auto flatInfo = kj::heapArray<const byte>(q[0].asBlob());
	auto inputStream = kj::heap<kj::ArrayInputStream>(flatInfo);
	auto messageReader = kj::heap<capnp::PackedMessageReader>(*inputStream);
	
	kj::Vector<Maybe<Own<ObjectDBEntry>>> entries;
	if(!shallow) {
		auto refs = targetBase.listOutgoingRefs.bind(id);
		while(refs.step()) {		
			if(refs[0].isNull()) {
				entries.add(nullptr);
				continue;
			}
			
			auto dbObject = parent -> open(refs[0], snapshotToUse);
			entries.add(mv(dbObject));
		}
	}
	
	// Defer wrapping behind enumeration so that
	// recursive wrap calls can use listOutgoingRefs
	// again.
	
	kj::Vector<Maybe<Own<ClientHook>>> rawCapTable;
	for(auto& maybeEntry : entries) {
		KJ_IF_MAYBE(ppEntry, maybeEntry) {
			rawCapTable.add(ClientHook::from(parent -> wrap(mv(*ppEntry))));
		} else {
			rawCapTable.add(nullptr);
		}
	}
	
	auto capTable = kj::heap<ReaderCapabilityTable>(rawCapTable.releaseAsArray());
	auto root = capTable -> imbue(messageReader -> getRoot<ObjectInfo>());
	
	return kj::heap<ObjectInfo::Reader>(root).attach(mv(messageReader), mv(inputStream), mv(flatInfo), mv(capTable));
}

Own<ObjectInfo::Reader> ObjectDBEntry::loadPreserved(bool shallow) {
	trySync(true);
	
	KJ_IF_MAYBE(pSnap, snapshot) {
		return doLoad(**pSnap, shallow);
	}
	KJ_FAIL_REQUIRE("loadPreserved() has no snapshot to load from. Ensure that hasPreserved() is true, e.g. by waiting for whenUpdated()", id);
}

bool ObjectDBEntry::exists() {
	auto& getRefcount = parent -> getRefcount;
	auto q = getRefcount.bind(this -> id);
	return q.step();
}

void ObjectDBEntry::restore() {
	KJ_REQUIRE(parent -> conn -> inTransaction());
	parent -> restoreObject(this -> id);
}

// class ObjectDB

// class ObjectDBEntry::Accessor

ObjectDBEntry::Accessor::Accessor(ObjectDBEntry& e) :
	ObjectInfo::Builder(nullptr),
	target(e.addRef())
{
	load();
}

ObjectDBEntry::Accessor::~Accessor() {
	if(!released && !ud.isUnwinding())
		save();
}

void ObjectDBEntry::Accessor::load() {
	KJ_REQUIRE(!target -> parent -> readOnly, "Can not perform write operations on target");
	auto inputReader = target -> doLoad(nullptr, false);
	
	auto outputPtr = capTable.imbue(messageBuilder.initRoot<AnyPointer>());
	outputPtr.setAs<ObjectInfo>(*inputReader);
	
	ObjectInfo::Builder::operator=(outputPtr.getAs<ObjectInfo>());
}

void ObjectDBEntry::Accessor::save() {
	KJ_REQUIRE(!target -> parent -> readOnly);
	
	// Decrease refcount of all previous outgoing references
	std::set<int64_t> idsToCheck;
	{
		auto outgoingRefs = target -> parent -> listOutgoingRefs.bind(target -> id);
		while(outgoingRefs.step()) {
			// Skip NULL
			if(outgoingRefs[0].isNull()) {
				continue;
			}
			
			int64_t refTarget = outgoingRefs[0];;
			target -> parent -> decRefcount(refTarget);
			idsToCheck.insert(refTarget);
		}
	}
	
	// Clear existing references
	target -> parent -> clearOutgoingRefs(target -> id);
	
	// Serialize the message into the database
	kj::VectorOutputStream os;
	capnp::writePackedMessage(os, messageBuilder);
	auto packed = os.getArray();
	target -> parent -> setInfo(target -> id, packed);
	
	// Iterate through the captured capabilities and store the links in the appropriate DB table
	kj::ArrayPtr<kj::Maybe<kj::Own<ClientHook>>> capTableData = capTable.getTable();
	for(auto i : kj::indices(capTableData)) {
		KJ_IF_MAYBE(pHook, capTableData[i]) {
			Capability::Client client((*pHook) -> addRef());
			auto unwrapped = target -> parent -> unwrap(client);
			
			KJ_REQUIRE(!unwrapped.is<Capability::Client>(), "Only immediate DB references and null capabilities may be used inside DBObject.");
			if(unwrapped.is<Own<ObjectDBEntry>>()) {
				auto& refTarget = unwrapped.get<Own<ObjectDBEntry>>();
			
				try {	
					auto rowid = target ->  parent -> insertRef.insert(target -> id, (int64_t) i, refTarget -> id);
					target -> parent -> incRefcount(refTarget -> id);
				} catch(...) {
					KJ_LOG(WARNING, "insertRef failed", target -> id, i, refTarget -> id, target -> exists(), refTarget -> exists(), asReader());
					throw;
				}
				
				continue;
			}
		}
		
		target -> parent -> insertRef(target -> id, (int64_t) i, nullptr);
	}
	
	// Check if we have capabilities without incoming refs
	for(int64_t id : idsToCheck) {
		target -> parent -> deleteIfOrphan(id);
	}
}

// class ObjectDBHook

ObjectDBHook::ObjectDBHook(Own<ObjectDBEntry> paramEntry) :
	entry(mv(paramEntry))
{
	auto cr = checkResolved();
	//entry -> trySync();
	
	//KJ_DBG(entry -> id);
	if(cr.is<Capability::Client>()) {
		//KJ_DBG("Created resolved hook");
		resolved = Resolved();
		inner = ClientHook::from(mv(cr.get<Capability::Client>()));
	} else if(cr.is<kj::Exception>()) {
		//KJ_DBG("Created broken hook");
		resolved = cr.get<kj::Exception>();
		inner = capnp::newBrokenCap(mv(cr.get<kj::Exception>()));;
	} else {
		//KJ_DBG("Created deferred hook");
		inner = capnp::newLocalPromiseClient(resolveTask());
	}
	// inner = capnp::newLocalPromiseClient(resolveTask());
	//KJ_DBG("Hook created", this);
}

ObjectDBHook::~ObjectDBHook()
{	
	// inner = capnp::newLocalPromiseClient(resolveTask());
	//KJ_DBG("Hook deleted", this);
}

Promise<Own<ClientHook>> ObjectDBHook::resolveTask() {
	return withODBBackoff([this]() -> Promise<Own<ClientHook>>  {
		auto cr = checkResolved();
		
		if(cr.is<Capability::Client>()) {
			resolved = Resolved();
			return ClientHook::from(mv(cr.get<Capability::Client>()));
		} else if(cr.is<kj::Exception>()) {
			resolved = cr.get<kj::Exception>();
			return capnp::newBrokenCap(mv(cr.get<kj::Exception>()));
		}
		
		return entry -> whenUpdated()
		.then([this]() {
			return resolveTask();
		});
	});
}

OneOf<decltype(nullptr), Capability::Client, kj::Exception> ObjectDBHook::checkResolved() {
	return createInterface(*entry, kj::Badge<ObjectDBHook>());
}

// ======================= Temporary files =====================

static inline capnp::CapabilityServerSet<Warehouse::File<>> tempFileSet;

/** Temporary file
 *
 * Represents a file that is held temporarily the server and can later be imported
 * into the database proper.
 */
struct TempFile : public Warehouse::File<>::Server {
	Promise<void> set(SetContext ctx);
	Promise<void> get(GetContext ctx);
	
	Promise<void> setAny(SetAnyContext ctx);
	Promise<void> getAny(GetAnyContext ctx);
	
	using Detached = Capability::Client;
	using Attached = Warehouse::File<>::Client;
	
	OneOf<Detached, Attached> storage = Detached(nullptr);
};

Warehouse::File<>::Client createTempFile();
Promise<Maybe<TempFile&>> checkTempFile(Warehouse::File<>::Client);

// class TempFile

Promise<void> TempFile::get(GetContext ctx) {
	if(storage.is<Detached>()) {
		ctx.initResults().setRef(storage.get<Detached>().castAs<DataRef<>>());
		return READY_NOW;
	}
	
	return ctx.tailCall(
		storage.get<Attached>().getRequest()
	);
}

Promise<void> TempFile::getAny(GetAnyContext ctx) {
	if(storage.is<Detached>()) {
		auto val = storage.get<Detached>();
		
		auto results = ctx.initResults();
		results.setAsGeneric(val.castAs<Warehouse::GenericObject>());
		results.setUnresolved();
		
		return READY_NOW;
	}
	
	return ctx.tailCall(
		storage.get<Attached>().getAnyRequest()
	);
}

Promise<void> TempFile::set(SetContext ctx) {
	if(storage.is<Detached>()) {
		storage.get<Detached>() = ctx.getParams().getRef();
		return READY_NOW;
	}
	
	auto tail = storage.get<Attached>().setRequest();
	tail.setRef(ctx.getParams().getRef());
	return ctx.tailCall(mv(tail));
}

Promise<void> TempFile::setAny(SetAnyContext ctx) {
	if(storage.is<Detached>()) {
		storage.get<Detached>() = ctx.getParams().getValue();
		return READY_NOW;
	}
	
	auto tail = storage.get<Attached>().setAnyRequest();
	tail.setValue(ctx.getParams().getValue());
	return ctx.tailCall(mv(tail));
}

Warehouse::File<>::Client createTempFile() {
	return tempFileSet.add(kj::heap<TempFile>());
}

Promise<Maybe<TempFile&>> checkTempFile(Warehouse::File<>::Client clt) {
	return tempFileSet.getLocalServer(clt)
	.then([](Maybe<Warehouse::File<>::Server&> server) mutable -> Maybe<TempFile&> {
		KJ_IF_MAYBE(pServer, server) {
			return static_cast<TempFile&>(*pServer);
		}
		return nullptr;
	});
}
	

// ======================= Import interfaces ===================

struct ImportContext;
struct DatarefDownloadProcess;
	
struct ImportTask {
	Maybe<Own<ObjectDBEntry>> entry;
	Promise<void> task = READY_NOW;
};

/** Context to import external capabilities.
 *
 * This class is responsible for helping convert non-native objects (currently
 * only DataRefs) to objects embedded in the database. This process requires 
 * careful coordination to ensure that the asynchronous tasks to fill in a
 * DataRef are ONLY started once the allocated IDs become permanent.
 
 * This class ensures this by managing the top-level transaction whenever an
 * import is required. An association between the created objects and their
 * import tasks is maintained, and once the top-level transaction completes,
 * the associated import tasks are started.
 */
struct ImportContext {
	ImportContext(ObjectDB&);
	~ImportContext();
	
	//! Create an (unresolved) object representing a future import.
	ImportTask import(Capability::Client target);
	ImportTask importIntoExisting(Capability::Client target, ObjectDBEntry& dst);
	
private:
	//! Kicks off import tasks, called by destructor.
	void scheduleImports();
	Promise<void> importTask(int64_t, Capability::Client target);
	
	Own<ObjectDB> parent;
	Own<db::Transaction> transaction;
	kj::UnwindDetector ud;
	
	struct PendingImport {
		int64_t id;
		Capability::Client src;
		Own<PromiseFulfiller<Promise<void>>> fulfiller;
	};
	std::list<PendingImport> pendingImports;
};

struct DatarefDownloadProcess : public internal::DownloadTask<Own<ObjectDBEntry>> {
	DatarefDownloadProcess(ObjectDB& db, int64_t id, DataRef<AnyPointer>::Client src);
	
	Promise<Maybe<Own<ObjectDBEntry>>> unwrap() override;
	Promise<Maybe<Own<ObjectDBEntry>>> useCached() override;
	Promise<void> receiveData(kj::ArrayPtr<const kj::byte> data) override;
	Promise<void> finishDownload() override;
	Promise<Own<ObjectDBEntry>> buildResult() override;
	
	ObjectDB& db;
	int64_t id;
	
	Own<BlobBuilder> builder;
	
	kj::Vector<Promise<void>> childImports;
	
	// bool firstData = true;
};

// class ImportContext

ImportContext::ImportContext(ObjectDB& parent) :
	parent(parent.addRef()), transaction(parent.writeTransaction())
{}

ImportContext::~ImportContext() {
	if(!ud.isUnwinding()) {
		transaction -> commit();
		scheduleImports();
	}
}

ImportTask ImportContext::import(Capability::Client target) {
	auto unwrapped = parent -> unwrap(target);
	
	if(unwrapped.is<decltype(nullptr)>())
		return ImportTask();
	
	if(unwrapped.is<Own<ObjectDBEntry>>()) {
		auto& entry = unwrapped.get<Own<ObjectDBEntry>>();
		
		// Check if object is still alive in DB
		if(entry -> exists()) {
			// Object is alive
			ImportTask result;
			result.entry = mv(entry);
			
			return result;
		}
		
		// Uh-oh. Object is dead. This can happen when
		// object is deleted by someone else while we
		// still have it open.
		
		// Check what it originally was
		auto preserved = entry -> loadPreserved();
		
		switch(preserved -> which()) {
			case ObjectInfo::UNRESOLVED:			
			case ObjectInfo::NULL_VALUE:
			case ObjectInfo::EXCEPTION:
			case ObjectInfo::DATA_REF:
				// The logic handles these correctly, restoring
				// the content from the last seen value (and
				// erroring out for unresolved objects as they
				// will never resolve).
				// The objects will be restored under a new ID,
				// but for immutable objects this doesn't matter.
				break;
			
			case ObjectInfo::LINK: {
				// Just import the links's target and apply
				// path shortening.
				return import(preserved -> getLink());
			}
			
			case ObjectInfo::FOLDER: {
				// Restore object under same ID
				entry -> restore();
				entry -> open() -> setFolder();
				
				// Iterate objects in folder
				kj::Vector<Promise<void>> subImports;
				
				auto q = entry -> getSnapshot().listFolderEntries.bind(entry -> id);
				while(q.step()) {
					kj::StringPtr name = q[0].asText();
					int64_t childId = q[1].asInt64();
					
					// Restore child
					ImportTask childImport = import(parent -> wrap(parent -> open(childId, entry -> getSnapshot())));
					
					KJ_IF_MAYBE(pChild, childImport.entry) {
						auto& childEntry = **pChild;
					
						parent -> incRefcount(childEntry.id);					
						parent -> updateFolderEntry(entry -> id, name, childEntry.id);
					}
					
					subImports.add(mv(childImport.task));
				}
				
				ImportTask result;
				result.entry = mv(entry);
				result.task = kj::joinPromises(subImports.releaseAsArray());
				
				return result;
			}
			case ObjectInfo::FILE: {
				// Restore object under same ID
				entry -> restore();
				
				ImportTask childImport = import(preserved -> getFile());
				
				KJ_IF_MAYBE(pChild, childImport.entry) {
					entry -> open() -> setFile(parent -> wrap(mv(*pChild)));
				} else {
					entry -> open() -> setFile(nullptr);
				}
				
				ImportTask result;
				result.entry = mv(entry);
				result.task = mv(childImport.task);
				
				return result;
			}
		}
	}
	
	auto newEntry = parent -> create();
	
	ImportTask result;
	result.entry = newEntry -> addRef();
	
	auto paf = kj::newPromiseAndFulfiller<Promise<void>>();
	result.task = mv(paf.promise);
	
	PendingImport import { newEntry -> id, target, mv(paf.fulfiller) };
	pendingImports.push_back(mv(import));
	
	return result;
}

ImportTask ImportContext::importIntoExisting(capnp::Capability::Client src, ObjectDBEntry& dst) {
	ImportTask result;
	result.entry = dst.addRef();
	
	auto paf = kj::newPromiseAndFulfiller<Promise<void>>();
	result.task = mv(paf.promise);
	
	PendingImport import { dst.id, src, mv(paf.fulfiller) };
	pendingImports.push_back(mv(import));
	
	return result;
}

void ImportContext::scheduleImports() {
	for(auto& import : pendingImports) {
		auto it = importTask(import.id, mv(import.src)).fork();
		import.fulfiller -> fulfill(it.addBranch());
		parent -> importTasks.add(it.addBranch());
	}
}

Promise<void> ImportContext::importTask(int64_t id, Capability::Client src) {
	// Note on lifetime: The task promises outlive ImportContext, but are
	// lifetime-bound to the parent DB. Therefore, we need to capture
	// parent by reference, id by value, and NOT capture "this".
	ObjectDB& myParent = *parent;
	
	return checkTempFile(src.castAs<Warehouse::File<>>())
	.then([src, id, &myParent](Maybe<TempFile&> maybeTempFile) mutable {
		// Step 1: If we have a temporary file, import that temp file into the
		// database.
		KJ_IF_MAYBE(pTempFile, maybeTempFile) {
			return withODBBackoff([src, id, &myParent, &tempFile = *pTempFile]() {
				ImportTask result;
				
				auto handle = myParent.open(id);
				
				using Attached = TempFile::Attached;
				using Detached = TempFile::Detached;
				
				// If the temp file is attached, we link to it.
				auto& storage = tempFile.storage;
				if(storage.is<Attached>()) {
					auto target = storage.get<Attached>();
					
					// Check that the target is from the same database
					bool compatible = !myParent.unwrap(target).is<Capability::Client>();
					KJ_REQUIRE(compatible, "Temp file was already attached to a different database");
					
					auto acc = handle -> open();
					acc -> setLink(target);
					
					return mv(result.task);
				} else {					
					{
						ImportContext ic(myParent);
						auto acc = handle -> open();
						result = ic.import(storage.get<Detached>());
						
						acc -> setFile(myParent.wrap(mv(result.entry)));
					}
					
					// If that all worked, register myself as backend to temp file
					storage.init<Attached>(myParent.wrap(myParent.open(id)).castAs<Warehouse::File<>>());
				}
				
				return mv(result.task);
			});
		}
	
		// Step 2: If it's not a temp file, attempt to download as DataRef
		auto dlp = kj::refcounted<DatarefDownloadProcess>(myParent, id, src.castAs<DataRef<AnyPointer>>());
		return dlp -> output().ignoreResult()
		
		// Check what type of exception it is
		.catch_([](kj::Exception&& e) -> Promise<void> {
			if(e.getType() == kj::Exception::Type::UNIMPLEMENTED)
				return KJ_EXCEPTION(FAILED, "The only foreign objects allowed in the database are DataRefs");
			
			return mv(e);
		});
	})
	
	// If it fails, store failure in database
	.catch_([&myParent, id](kj::Exception&& e) mutable {
		return withODBBackoff([&myParent, id, e = mv(e)]() {
			auto t = myParent.writeTransaction();
			
			// Delete partially written results
			auto q = myParent.getBlob.bind(id);
			
			if(!q.step()) {
				// Object was deleted, just return
				return;
			}
			
			if(!q[0].isNull()) {
				int64_t blobId = q[0];
				
				myParent.setBlob(id, nullptr);
				myParent.blobStore -> get(blobId) -> decRef();
			}
			
			// Write exception into object record
			myParent.open(id)
			-> open()
			-> setException(
				toProto(e)
			);
		});
	});
}

// class DatarefDownloadProcess

DatarefDownloadProcess::DatarefDownloadProcess(ObjectDB& db, int64_t id, DataRef<AnyPointer>::Client src) :
	DownloadTask(src, Context()),
	db(db), id(id)
{}

Promise<Maybe<Own<ObjectDBEntry>>> DatarefDownloadProcess::unwrap() {
	return withODBBackoff([this]() mutable -> Maybe<Own<ObjectDBEntry>> {
		auto unwrapResult = db.unwrap(src);
		
		if(unwrapResult.is<Own<ObjectDBEntry>>()) {
			auto dst = db.open(id);
			
			if(unwrapResult.get<Own<ObjectDBEntry>>() -> exists()) {
				auto acc = dst -> open();
				acc -> setLink(src);
			
				return mv(dst);
			}
		} else if(unwrapResult.is<decltype(nullptr)>()) {
			auto dst = db.open(id);
			auto acc = dst -> open();
			acc -> setNullValue();
			
			return mv(dst);
		}
		return nullptr;
	});
}

Promise<Maybe<Own<ObjectDBEntry>>> DatarefDownloadProcess::useCached() {
	// Give nested capabilities some time to progress their resolution
	return withODBBackoff([this]() mutable -> Maybe<Own<ObjectDBEntry>> {		
		ImportContext ic(db);
		auto dst = db.open(id);
		auto acc = dst -> open();
		
		Maybe<capnp::Capability::Client> replaces;
		if(acc -> isUnresolved() && acc -> getUnresolved().hasPreviousValue())
			replaces = acc -> getUnresolved().getPreviousValue();
		
		// Initialize target to DataRef
		auto ref = acc -> initDataRef();
		ref.setMetadata(metadata);
		
		// KJ_DBG(metadata);
		
		auto capTableOut = ref.initCapTable(capTable.size());
		for(auto i : kj::indices(capTable)) {
			auto importTask = ic.import(capTable[i]);
			
			// Propagate unresolved to children
			KJ_IF_MAYBE(ppEntry, importTask.entry) {
				auto targetAcc = (**ppEntry).open();
				if(targetAcc -> isUnresolved() && !targetAcc -> getUnresolved().hasPreviousValue()) {
					KJ_IF_MAYBE(pReplaces, replaces) {
						targetAcc -> getUnresolved().setPreviousValue(*pReplaces);
					}
				}
			}
			capTableOut.set(i, db.wrap(mv(importTask.entry)));
			childImports.add(mv(importTask.task));
		}
		
		// Check if we have hash in blob store
		KJ_IF_MAYBE(pBlob, db.blobStore -> find(metadata.getDataHash())) {
			db.setBlob(id, (**pBlob).getId());
			(**pBlob).incRef();
			
			ref.getDownloadStatus().setFinished();
			
			return mv(dst);
		} else {
			ref.getDownloadStatus().setDownloading();
		}
		
		// Transfer data
		ref.getMetadata().setDataHash(nullptr);
		
		// Allocate blob builder
		constexpr size_t CHUNK_SIZE = 1024 * 1024;
		builder = db.blobStore -> create(CHUNK_SIZE);
		
		// Set blob id under construction so that the blob gets deleted with
		// the parent object.
		db.setBlob(id, builder -> getBlobUnderConstruction() -> getId());
		
		return nullptr;
	}).then([this](Maybe<Own<ObjectDBEntry>> e) -> Promise<Maybe<Own<ObjectDBEntry>>> {
		if(e == nullptr)
			return mv(e);
		
		return kj::joinPromises(childImports.releaseAsArray())
		.then([e = mv(e)]() mutable {
			return mv(e);
		});
	});
}
	
Promise<void> DatarefDownloadProcess::receiveData(kj::ArrayPtr<const kj::byte> data) {
	return withODBBackoff([this, data]() mutable -> Promise<void> {
		KJ_REQUIRE(builder.get() != nullptr);
		
		return getActiveThread().worker().executeAsync([this, data]() {
			return builder -> tryConsume(data);
		})
		.then([this, data](bool consumed) -> Promise<void> {
			if(consumed)
				return READY_NOW;
			
			db.writeLock();
			builder -> flush();
			return receiveData(data);
		});
		
		/*
		// BlobBuilder::write() does not go inside a transaction
		// Still, enable the global batch transaction
		db.writeLock();
		builder -> write(data.begin(), data.size());
		
		return READY_NOW;*/
	});
}

Promise<void> DatarefDownloadProcess::finishDownload() {
	return withODBBackoff([this]() mutable -> Promise<void> {
		KJ_REQUIRE(builder.get() != nullptr);
		
		auto t = db.writeTransaction();
		auto finishedBlob = builder -> finish();
		KJ_REQUIRE(finishedBlob -> isFinished(), "Blob unfinished after finish() call");
		db.setBlob(id, finishedBlob -> getId());
		
		// Mark ref as done and store hash
		{
			auto acc = db.open(id) -> open();
			auto ref = acc -> getDataRef();
			
			ref.getDownloadStatus().setFinished();
			ref.getMetadata().setDataHash(finishedBlob -> getHash());
		}
		
		return READY_NOW;
	});
}

Promise<Own<ObjectDBEntry>> DatarefDownloadProcess::buildResult() {
	return kj::joinPromises(childImports.releaseAsArray())
	.then([this]() {
		return db.open(id);
	});
}	

// ======================= Export interfaces ===================
	

// ======================= Access interfaces ===================

struct FileInterface;
struct FolderInterface;
struct DataRefInterface;
struct TransmissionProcess;

//! Interface class for file objects
struct FileInterface : public Warehouse::File<AnyPointer>::Server {
	// Interface
	Promise<void> set(SetContext ctx);
	Promise<void> get(GetContext ctx);
	
	Promise<void> setAny(SetAnyContext ctx);
	Promise<void> getAny(GetAnyContext ctx);
	
	// Implementation
	FileInterface(Own<ObjectDBEntry>&& object, kj::Badge<ObjectDBHook>) :
		object(mv(object))
	{}
	
	Promise<void> setImpl(Capability::Client);
	
	// Members
	Own<ObjectDBEntry> object;
};

//! Interface class for folder objects
struct FolderInterface : public Warehouse::Folder::Server {
	// Interface
	Promise<void> ls(LsContext ctx) override;
	Promise<void> getAll(GetAllContext ctx) override;
	Promise<void> get(GetContext ctx) override;
	Promise<void> put(PutContext ctx) override;
	Promise<void> rm(RmContext ctx) override;
	Promise<void> mkdir(MkdirContext ctx) override;
	Promise<void> createFile(CreateFileContext ctx) override;
	Promise<void> freeze(FreezeContext ctx) override;
	Promise<void> exportGraph(ExportGraphContext ctx) override;
	Promise<void> importGraph(ImportGraphContext ctx) override;
	Promise<void> deepCopy(DeepCopyContext ctx) override;

	// Implementation
	FolderInterface(Own<ObjectDBEntry>&& object, kj::Badge<ObjectDBHook>) :
		object(mv(object))
	{}
		
	//! Locates a directory for modification
	Own<ObjectDBEntry> locateWriteDir(kj::PathPtr path, bool create);
	
	//! Reads an object relative to this one. Uses the preserved snapshot of this folder.
	Own<ObjectDBEntry> locate(kj::PathPtr path);
	
	Promise<void> freezeImpl(Own<ObjectDBEntry> e, Warehouse::FrozenFolder::Builder out);
	Promise<void> freezeImpl(Own<ObjectDBEntry> e, Warehouse::FrozenEntry::Builder out);
	
	// Members
	Own<ObjectDBEntry> object;
};

struct DataRefInterface : public DataRef<AnyPointer>::Server {
	// DataRef interface
	Promise<void> metaAndCapTable(MetaAndCapTableContext ctx) override;
	Promise<void> rawBytes(RawBytesContext ctx) override;
	Promise<void> transmit(TransmitContext ctx) override;

	// Implementation
	DataRefInterface(Own<ObjectDBEntry>&& object, kj::Badge<ObjectDBHook>) :
		object(mv(object))
	{}
	
	// Members
	Own<ObjectDBEntry> object;
};

struct TransmissionProcess {
	constexpr static inline size_t CHUNK_SIZE = 1024 * 1024;
	
	Own<BlobReader> reader;
	
	DataRef<capnp::AnyPointer>::Receiver::Client receiver;
	size_t start;
	size_t end;
	
	// Array<byte> buffer;
	
	TransmissionProcess(Own<BlobReader>&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end);
	
	Promise<void> run();
	Promise<void> transmit(size_t chunkStart);
};

// class FileInterface

Promise<void> FileInterface::setImpl(Capability::Client clt) {
	return withODBBackoff([this, clt = mv(clt)]() mutable {
		ImportTask it;
		{
			ImportContext ic(object -> getDb());
			
			auto acc = object -> open();
			
			it = ic.import(clt);
			
			KJ_IF_MAYBE(pImported, it.entry) {
				{
					auto acc2 = (**pImported).open();
					
					if(acc2 -> isUnresolved() && !acc2 -> getUnresolved().hasPreviousValue()) {
						acc2 -> getUnresolved().setPreviousValue(acc -> getFile());
					}
				}
				acc -> setFile(object -> getDb().wrap(mv(*pImported)));
			} else {
				acc -> setFile(nullptr);
			}
		}
		
		return mv(it.task);
	})
	.then([this]() {
		return object -> getDb().writeBarrier();
	});
}

Promise<void> FileInterface::getAny(GetAnyContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto data = object -> loadPreserved();
		object -> getDb().exportStoredObject(data -> getFile(), ctx.initResults());
	});
}

Promise<void> FileInterface::get(GetContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto data = object -> loadPreserved();
		ctx.getResults().setRef(data -> getFile().castAs<DataRef<AnyPointer>>());
	});
}

Promise<void> FileInterface::set(SetContext ctx) {
	return setImpl(ctx.getParams().getRef());
}

Promise<void> FileInterface::setAny(SetAnyContext ctx) {
	return setImpl(ctx.getParams().getValue());
}

// class FolderInterface

Promise<void> FolderInterface::ls(LsContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto data = locate(kj::Path::parse(ctx.getParams().getPath()));
		data -> trySync(true);
		
		auto& snap = data -> getSnapshot();
		
		auto q = snap.listFolderEntries.bind(data -> id);
		
		std::list<kj::String> tmp;
		while(q.step()) {
			tmp.push_back(kj::heapString(q[0].asText()));
		}
		
		auto out = ctx.getResults().initEntries(tmp.size());
		for(auto i : kj::indices(out)) {
			out.set(i, *tmp.begin());
			tmp.pop_front();
		}
	});
}

Promise<void> FolderInterface::getAll(GetAllContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto data = locate(kj::Path::parse(ctx.getParams().getPath()));
		data -> trySync(true);
		
		auto& snap = data -> getSnapshot();
		
		auto q = snap.listFolderEntries.bind(data -> id);
		
		std::list<kj::String> tmp;
		std::list<int64_t> tmp2;
		while(q.step()) {
			tmp.push_back(kj::heapString(q[0].asText()));
			tmp2.push_back(q[1].asInt64());
		}
		
		auto out = ctx.getResults().initEntries(tmp.size());
		for(auto i : kj::indices(out)) {
			auto eOut = out[i];
			eOut.setName(*tmp.begin());
			
			auto entry = object -> getDb().open(*tmp2.begin(), snap);
			auto wrapped = object -> getDb().wrap(mv(entry));
			
			object -> getDb().exportStoredObject(wrapped, eOut.initValue());
			
			tmp.pop_front();
			tmp2.pop_front();
		}
	});
}

Promise<void> FolderInterface::get(GetContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto target = locate(kj::Path::parse(ctx.getParams().getPath()));
		auto wrapped = object -> getDb().wrap(mv(target));
		
		object -> getDb().exportStoredObject(wrapped, ctx.initResults());
	});
}

Promise<void> FolderInterface::put(PutContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		ImportTask importTask;
		auto& db = object -> getDb();
		
		kj::Path path = kj::Path::parse(ctx.getParams().getPath());
		
		{
			ImportContext ic(db);
			
			auto parentFolder = locateWriteDir(path.parent(), true);
			auto& filename = path.basename()[0];
			
			// Decrease old refcount if present
			Maybe<int64_t> oldId;
			auto oldIdQuery = db.getFolderEntry.bind(parentFolder -> id, filename.asPtr());
			if(oldIdQuery.step()) {
				oldId = oldIdQuery[0].asInt64();
			}
			
			importTask = ic.import(ctx.getParams().getValue());
			
			KJ_IF_MAYBE(pEntry, importTask.entry) {				
				// If we have a previous object, register it in the unitialized object
				KJ_IF_MAYBE(pOldId, oldId) {
					auto oldEntry = db.wrap(db.open(*pOldId));
					
					auto accNew = (**pEntry).open();
					if(accNew -> isUnresolved() && !accNew -> getUnresolved().hasPreviousValue()) {
						accNew -> getUnresolved().setPreviousValue(oldEntry);
					}
				}
				
				// Update object
				int64_t id = (**pEntry).id;
				
				db.incRefcount(id);
				db.updateFolderEntry(parentFolder -> id, filename.asPtr(), id);
			} else {
				db.deleteFolderEntry(parentFolder -> id, filename.asPtr());
			}
			
			KJ_IF_MAYBE(pOldId, oldId) {
				db.decRefcount(*pOldId);
				db.deleteIfOrphan(*pOldId);
			}
		}
		
		return importTask.task
		.then([&db]() { return db.writeBarrier(); })
		.then([&db, e = mv(importTask.entry), ctx]() mutable {
			KJ_IF_MAYBE(pEntry, e) {
				// Export saved object to result
				db.exportStoredObject(db.wrap((**pEntry).addRef()), ctx.initResults());
			} else {
				ctx.initResults().setNullValue();
			}
		});
	});
}

Promise<void> FolderInterface::rm(RmContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto& db = object -> getDb();
		
		kj::Path path = kj::Path::parse(ctx.getParams().getPath());
		
		{
			auto t = db.writeTransaction();
			
			auto parentFolder = locateWriteDir(path.parent(), false);
			auto& filename = path.basename()[0];
			
			// Decrease old refcount if present
			Maybe<int64_t> oldId;
			auto oldIdQuery = db.getFolderEntry.bind(parentFolder -> id, filename.asPtr());
			if(oldIdQuery.step()) {
				oldId = oldIdQuery[0].asInt64();
			}
			
			db.deleteFolderEntry(parentFolder -> id, filename.asPtr());
			
			KJ_IF_MAYBE(pOldId, oldId) {
				db.decRefcount(*pOldId);
				db.deleteIfOrphan(*pOldId);
			}
		}
		
		return db.writeBarrier();
	});
}

Promise<void> FolderInterface::mkdir(MkdirContext ctx) {
	return withODBBackoff([this, ctx] () mutable -> Promise<void> {
		kj::Path path = kj::Path::parse(ctx.getParams().getPath());
		
		auto result = locateWriteDir(path, true);
		ctx.initResults().setFolder(object -> getDb().wrap(mv(result)).castAs<Warehouse::Folder>());
		
		return object -> getDb().writeBarrier();
	});
}

Promise<void> FolderInterface::createFile(CreateFileContext ctx) {
	// Temporary file creation
	if(ctx.getParams().getPath().size() == 0) {
		ctx.initResults().setFile(createTempFile());
		return READY_NOW;
	}
	
	return withODBBackoff([this, ctx] () mutable -> Promise<void> {
		auto& db = object -> getDb();
		
		{
			auto t = db.writeTransaction();
			
			kj::Path path = kj::Path::parse(ctx.getParams().getPath());
			
			auto parentFolder = locateWriteDir(path.parent(), true);
			auto& filename = path.basename()[0];
			
			// Decrease old refcount if present
			Maybe<int64_t> oldId;
			auto oldIdQuery = db.getFolderEntry.bind(parentFolder -> id, filename.asPtr());
			if(oldIdQuery.step()) {
				oldId = oldIdQuery[0].asInt64();
			}
			
			auto newFile = db.create();
			newFile -> open() -> setFile(nullptr);
						
			db.incRefcount(newFile -> id);
			db.updateFolderEntry(parentFolder -> id, filename.asPtr(), newFile -> id);
			
			ctx.initResults().setFile(db.wrap(mv(newFile)).castAs<Warehouse::File<>>());
			
			KJ_IF_MAYBE(pOldId, oldId) {
				db.decRefcount(*pOldId);
				db.deleteIfOrphan(*pOldId);
			}
		}
		
		return db.writeBarrier();
	});
}

Promise<void> FolderInterface::freeze(FreezeContext ctx) {
	Temporary<Warehouse::FrozenFolder> target;
	
	Promise<void> doFreeze = freezeImpl(locate(kj::Path::parse(ctx.getParams().getPath())), target);
	
	return doFreeze
	.then([target = mv(target), ctx]() mutable {
		auto ref = getActiveThread().dataService().publish(target.asReader());
		ctx.initResults().setRef(mv(ref));
	});
}

Promise<void> FolderInterface::exportGraph(ExportGraphContext ctx) {
	return withODBBackoff([this, ctx] () mutable {
		auto entry = locate(kj::Path::parse(ctx.getParams().getPath()));
		
		auto& snapshot = entry -> getSnapshot();
		
		// Collect mappings from DB id to section IDs
		kj::HashMap<int64_t, uint64_t> idMap;
		kj::Vector<int64_t> ids;
		{
			std::list<int64_t> queue;
			uint64_t counter = 0;
			
			queue.push_back(entry -> id);
			
			while(!queue.empty()) {
				int64_t front = *queue.begin();
				queue.pop_front();
				
				if(idMap.find(front) != nullptr)
					continue;
				
				idMap.insert(front, counter++);
				ids.add(front);
				
				auto q1 = snapshot.listOutgoingRefs.bind(front);
				while(q1.step()) {
					if(q1[0].isNull()) continue;
					
					queue.push_back(q1[0].asInt64());
				}
				
				auto q2 = snapshot.listFolderEntries.bind(front);
				while(q2.step()) {
					queue.push_back(q2[1].asInt64());
				}
			}
		}
		
		KJ_REQUIRE(ids.size() <= ((uint64_t) 1) << 48, "Graph size too large for storage");
		
		constexpr uint64_t MAX_LIST_SIZE = 1 << 29 - 1;
		const uint64_t nLists = ids.size() / MAX_LIST_SIZE + 1;
		
		Temporary<Warehouse::ObjectGraph> result;
		auto entryLists = result.initObjects(nLists);
		
		for(auto i : kj::indices(entryLists)) {
			uint64_t listStart = MAX_LIST_SIZE * i;
			uint64_t listEnd = kj::min(listStart + MAX_LIST_SIZE, ids.size());
			
			if(listEnd > listStart) {
				entryLists.init(i, listEnd - listStart);
			}
		}
		
		for(auto i : kj::indices(ids)) {
			int64_t id = ids[i];
			
			uint64_t addr1 = i / MAX_LIST_SIZE;
			uint64_t addr2 = i - addr1 * MAX_LIST_SIZE;
			
			auto outEntry = entryLists[addr1][addr2];
			
			auto& db = entry -> getDb();
			auto current = db.open(id, snapshot);
			auto preserved = current -> loadPreserved();
			
			auto exportObject = [&](capnp::Capability::Client obj, Warehouse::ObjectGraph::ObjectRef::Builder out) {
				auto unwrapped = db.unwrap(obj);
				if(unwrapped.is<decltype(nullptr)>()) {
					out.setNull();
					return;
				}
				
				KJ_REQUIRE(unwrapped.is<Own<ObjectDBEntry>>(), "Internal error: DB yielded un-unwrappable object during export");
				KJ_IF_MAYBE(pTargetId, idMap.find(unwrapped.get<Own<ObjectDBEntry>>() -> id)) {
					out.setObject(*pTargetId);
				} else {
					KJ_FAIL_REQUIRE("Could not resolve ID of linked object");
				}
			};
			
			switch(preserved -> which()) {
				case ObjectInfo::UNRESOLVED:
					outEntry.setUnresolved();
					break;
				
				case ObjectInfo::NULL_VALUE:
					outEntry.setNullValue();
					break;
				
				case ObjectInfo::EXCEPTION:
					outEntry.setException(preserved -> getException());
					break;
				
				case ObjectInfo::LINK:
					outEntry.setLink(db.unwrapValid(preserved -> getLink()) -> id);
					break;
					
				case ObjectInfo::DATA_REF: {					
					auto refInfo = outEntry.initDataRef();
					refInfo.setData(overrideRefs(db.wrap(current -> addRef()).castAs<DataRef<AnyPointer>>(), nullptr));
					
					auto capTableIn = preserved -> getDataRef().getCapTable();
					auto capTableOut = refInfo.initRefs(capTableIn.size());
					
					for(auto i : kj::indices(capTableIn)) {
						exportObject(capTableIn[i], capTableOut[i]);
					}
					
					break;
				}
				
				case ObjectInfo::FOLDER: {
					kj::Vector<kj::Tuple<kj::String, uint64_t>> entries;
					
					auto q = snapshot.listFolderEntries.bind(id);
					while(q.step()) {
						auto name = q[0].asText();
						
						KJ_IF_MAYBE(pId, idMap.find(q[1].asInt64())) {
							entries.add(kj::tuple(kj::heapString(name), *pId));
						} else {
							KJ_FAIL_REQUIRE("Internal error: Id lookup failed");
						}
					}
					
					auto entriesOut = outEntry.initFolder(entries.size());
					for(auto iRef : kj::indices(entries)) {
						entriesOut[iRef].setName(kj::get<0>(entries[iRef]));
						entriesOut[iRef].setObject(kj::get<1>(entries[iRef]));
					}
					break;
				}
				
				case ObjectInfo::FILE: {
					exportObject(preserved -> getFile(), outEntry.initFile());
					break;
				}
			}
		}
		
		ctx.initResults().setGraph(
			getActiveThread().dataService().publish(mv(result))
		);
	});
}

Promise<void> FolderInterface::importGraph(ImportGraphContext ctx) {
	return getActiveThread().dataService().download(ctx.getParams().getGraph())
	.then([this, ctx](auto graph) {
		return withODBBackoff([this, ctx, graph = mv(graph)] () mutable {
			auto& db = object -> getDb();
			
			{
				auto t = db.writeTransaction();
				
				kj::Path path = kj::Path::parse(ctx.getParams().getPath());
				
				auto parentFolder = locateWriteDir(path.parent(), true);
				auto& objectName = path.basename()[0];
				
				// Decrease old refcount if present
				Maybe<int64_t> oldId;
				auto oldIdQuery = db.getFolderEntry.bind(parentFolder -> id, objectName.asPtr());
				if(oldIdQuery.step()) {
					oldId = oldIdQuery[0].asInt64();
				}
				
				// Prepare import of graph
				auto objects = graph.get().getObjects();
				kj::HashMap<uint64_t, Own<ObjectDBEntry>> importMap;
				ImportContext ic(db);
				kj::Vector<Promise<void>> importTasks;
				
				// Calculate starting points of all lists
				auto startAddresses = kj::heapArray<uint64_t>(objects.size());
				{
					uint64_t offset = 0;
					for(auto i : kj::indices(objects)) {
						startAddresses[i] = offset;
						offset += objects[i].size();
					}
				}
				
				kj::Function<ObjectDBEntry& (uint64_t)> import = [&](uint64_t id) -> ObjectDBEntry& {
					// Check if object already allocated
					KJ_IF_MAYBE(pEntry, importMap.find(id)) {
						return **pEntry;
					}
					
					using Object = Warehouse::ObjectGraph::Object;	
					size_t listIdx = std::upper_bound(startAddresses.begin(), startAddresses.end(), id) - startAddresses.begin() - 1;
					Object::Reader object = objects[listIdx][id - startAddresses[listIdx]];
					
					// Helper to import object references
					auto importRef = [&](Warehouse::ObjectGraph::ObjectRef::Reader ref) -> Maybe<Own<ObjectDBEntry>> {
						if(ref.isNull()) return nullptr;
						if(ref.isObject()) return import(ref.getObject()).addRef();
						KJ_FAIL_REQUIRE("Unknown ref type");
					};
					
					// Allocate object
					Own<ObjectDBEntry> dst = db.create();
					importMap.insert(id, dst -> addRef());
									
					switch(object.which()) {
						case Object::UNRESOLVED:
							dst -> open() -> setException(toProto(KJ_EXCEPTION(FAILED, "Object was unresolved at export time")));
							break;
						case Object::NULL_VALUE:
							dst -> open() -> setNullValue();
							break;
						case Object::EXCEPTION:
							dst -> open() -> setException(object.getException());
							break;
						case Object::DATA_REF: {
							auto asDR = object.getDataRef();
							auto refsIn = asDR.getRefs();
							auto refBuilder = kj::heapArrayBuilder<capnp::Capability::Client>(refsIn.size());
							
							for(auto refIn : refsIn) {
								refBuilder.add(db.wrap(importRef(refIn)));
							}
							
							// We need to find a place to attach the objects so that they
							// stay alive. Since we have an unfinished DataRef, we can use
							// that as a housing place.
							{
								auto acc = dst -> open();
								auto ref = acc -> initDataRef();
								auto ct = ref.initCapTable(refBuilder.size());
								for(auto i : kj::indices(refBuilder))
									ct.set(i, refBuilder[i]);
							}
													
							auto importSrc = overrideRefs(asDR.getData(), refBuilder.finish());
							
							auto task = ic.importIntoExisting(importSrc, *dst);
							importTasks.add(mv(task.task));
							break;
						}
						case Object::FOLDER: {
							dst -> open() -> setFolder();
													
							for(auto e : object.getFolder()) {
								// Check that entry does not exist (otherwise we get inconsistent refcount)
								KJ_REQUIRE(!db.getFolderEntry.bind(dst -> id, e.getName()).step(), "Duplicate folder names in input");
								
								// Attach child object
								auto& child = import(e.getObject());
								db.incRefcount(child.id);
								db.updateFolderEntry(dst -> id, e.getName(), child.id);
							}
							
							break;
						}
						case Object::FILE:
							dst -> open() -> setFile(db.wrap(importRef(object.getFile())));
							break;
						case Object::LINK:
							dst -> open() -> setLink(db.wrap(import(object.getLink()).addRef()));
							break;
						default:
							KJ_FAIL_REQUIRE("Import failure: Unknown object type");
					}
					
					return *dst;
				};
				
				auto& newObject = import(ctx.getParams().getRoot());
				
				db.incRefcount(newObject.id);
				db.updateFolderEntry(parentFolder -> id, objectName.asPtr(), newObject.id);
				
				KJ_IF_MAYBE(pOldId, oldId) {
					db.decRefcount(*pOldId);
					db.deleteIfOrphan(*pOldId);
				}
				
				db.exportStoredObject(db.wrap(newObject.addRef()), ctx.initResults());
				return kj::joinPromisesFailFast(importTasks.releaseAsArray())
				.then([&db]() mutable {
					return db.writeBarrier();
				});
			}
		});
	});
}

Promise<void> FolderInterface::deepCopy(DeepCopyContext ctx) {
	auto exportReq = thisCap().exportGraphRequest();
	exportReq.setPath(ctx.getParams().getSrcPath());
	
	return exportReq.send().then([this, ctx](auto graphResponse) mutable {
		auto importReq = thisCap().importGraphRequest();
		importReq.setPath(ctx.getParams().getDstPath());
		importReq.setGraph(graphResponse.getGraph());
		
		return ctx.tailCall(mv(importReq));
	});
}

Own<ObjectDBEntry> FolderInterface::locateWriteDir(kj::PathPtr path, bool create) {
	auto& db = object -> getDb();
	KJ_REQUIRE(db.conn -> inTransaction(), "INTERNAL ERROR: locateWriteDir must be called inside transaction");
	
	Own<ObjectDBEntry> current = object -> addRef();
	
	for(const auto& folderName : path) {
		auto acc = current -> open();
		KJ_REQUIRE(acc -> isFolder(), "Trying to access child object of non-folder", path, folderName);
				
		auto q = db.getFolderEntry.bind(current -> id, folderName.asPtr());
		if(q.step()) {
			current = db.open(q[0]);
		} else {
			auto newFolder = object -> getDb().create();
			auto newFolderAcc = newFolder -> open();
			newFolderAcc -> setFolder();
			newFolderAcc -> release(true);
			
			db.updateFolderEntry(current -> id, folderName.asPtr(), newFolder -> id);
			db.incRefcount(newFolder -> id);
			
			current = mv(newFolder);
		}
	}
	
	auto acc = current -> open();
	KJ_REQUIRE(acc -> isFolder(), "Trying to modify non-folder as folder");
	acc -> release(false);
	
	return current;
	
}

Own<ObjectDBEntry> FolderInterface::locate(kj::PathPtr path) {
	KJ_REQUIRE(object -> hasPreserved(), "Internal error: Folder has no preserved state");
	
	Own<ObjectDBEntry> current = object -> addRef();
	for(const auto& entryName : path) {
		current -> trySync(true);
		
		auto acc = current -> loadPreserved();
		
		if(acc -> isFolder()) {
			ObjectDBSnapshot& snap = current -> getSnapshot();
			auto q = snap.getFolderEntry.bind(current -> id, entryName.asPtr());
			KJ_REQUIRE(q.step(), "Entry not found", path, entryName);
		
			current = object -> getDb().open(q[0], snap);
		} else if(acc -> isFile()) {
			KJ_REQUIRE(entryName == "contents");
			current = object -> getDb().unwrapValid(acc -> getFile());
		} else if(acc -> isDataRef()) {
			auto ref = acc -> getDataRef();
			
			KJ_IF_MAYBE(pIndex, entryName.tryParseAs<uint64_t>()) {
				current = object -> getDb().unwrapValid(ref.getCapTable()[*pIndex]);
			} else {
				KJ_FAIL_REQUIRE("Child objects of DataRefs must be accessed by unsigned int index", entryName);
			}
		} else if(acc -> isLink()) {
			current = object -> getDb().unwrapValid(acc -> getLink());
		} else {
			KJ_FAIL_REQUIRE("The requested object can not provide child elements");
		}
	}
	
	current -> trySync(true);
	
	return current;
}

Promise<void> FolderInterface::freezeImpl(Own<ObjectDBEntry> e, Warehouse::FrozenFolder::Builder out) {
	return withODBBackoff([this, e = mv(e), out]() mutable {
		ObjectDBSnapshot& snap = e -> getSnapshot();
		
		std::list<kj::String> tmp;
		std::list<int64_t> tmp2;
		
		auto q = snap.listFolderEntries.bind(e -> id);
		while(q.step()) {
			tmp.push_back(kj::heapString(q[0].asText()));
			tmp2.push_back(q[1].asInt64());
		}
		
		auto promiseBuilder = kj::heapArrayBuilder<Promise<void>>(tmp.size());
		
		auto entries = out.initEntries(tmp.size());
		for(auto i : kj::indices(entries)) {
			entries[i].setName(*tmp.begin());
			promiseBuilder.add(freezeImpl(
				e -> getDb().open(*tmp2.begin(), snap),
				entries[i].initValue()
			));
			
			tmp.pop_front();
			tmp2.pop_front();
		}
		
		return kj::joinPromises(promiseBuilder.finish());
	});
}

Promise<void> FolderInterface::freezeImpl(Own<ObjectDBEntry> e, Warehouse::FrozenEntry::Builder out) {
	return withODBBackoff([this, e = mv(e), out]() mutable -> Promise<void> {
		auto data = e -> loadPreserved();
		
		switch(data -> which()) {
			case ObjectInfo::UNRESOLVED:
				out.setUnavailable();
				return READY_NOW;
			case ObjectInfo::NULL_VALUE:
				out.setUnavailable();
				return READY_NOW;
			case ObjectInfo::EXCEPTION:
				out.setUnavailable();
				return READY_NOW;
			case ObjectInfo::LINK:
				return freezeImpl(e -> getDb().unwrapValid(data -> getLink()), out);
			case ObjectInfo::DATA_REF:
				out.setDataRef(e -> getDb().wrap(e -> addRef()).castAs<DataRef<>>());
				return READY_NOW;
			case ObjectInfo::FOLDER:
				return freezeImpl(e -> addRef(), out.initFolder());
			case ObjectInfo::FILE: {
				if(!data -> hasFile()) {
					out.initFile().setNullValue();
					return READY_NOW;
				} else {
					return freezeImpl(e -> getDb().unwrapValid(data -> getFile()), out.initFile().initValue());
				}
			}
			default:
				KJ_FAIL_REQUIRE("Unknown object type");
		}
	});
}

// class DataRefInterface

Promise<void> DataRefInterface::metaAndCapTable(MetaAndCapTableContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto data = object -> loadPreserved();
		
		auto results = ctx.initResults();
		auto ref = data -> getDataRef();
		
		results.setMetadata(ref.getMetadata());
		results.setTable(ref.getCapTable());
	});
}

Promise<void> DataRefInterface::rawBytes(RawBytesContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		const uint64_t start = ctx.getParams().getStart();
		const uint64_t end = ctx.getParams().getEnd();
		
		auto snapshot = object -> getSnapshot().addRef();
		
		// Query blob id
		auto blobIdQuery = snapshot -> getBlob.bind(object -> id);
		KJ_REQUIRE(blobIdQuery.step(), "Internal error: Object deleted", object -> id);
		
		// Open blob
		Own<Blob> blob = snapshot -> blobStore -> get(blobIdQuery[0]);
		Own<kj::InputStream> reader = blob -> open();
		reader -> skip(start);
		
		// Decompress into output buffer
		auto data = ctx.getResults().initData(end - start);
		reader -> read(data.begin(), data.size());
	});
}

Promise<void> DataRefInterface::transmit(TransmitContext ctx) {
	return withODBBackoff([this, ctx]() mutable {
		auto params = ctx.getParams();
	
		auto snapshot = object -> getSnapshot().addRef();
		
		auto blobIdQuery = snapshot -> getBlob.bind(object -> id);
		KJ_REQUIRE(blobIdQuery.step(), "Internal error: Object deleted", object -> id);
		
		// Open blob
		Own<Blob> blob = snapshot -> blobStore -> get(blobIdQuery[0]);
		Own<BlobReader> reader = blob -> open();
		
		// Create transmission process
		auto transProc = heapHeld<TransmissionProcess>(mv(reader), params.getReceiver(), params.getStart(), params.getEnd());
		return transProc -> run().attach(transProc.x());
	});
}

// class TransmissionProcess
	
TransmissionProcess::TransmissionProcess(Own<BlobReader>&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end) :
	reader(mv(reader)),
	receiver(mv(receiver)),
	// buffer(kj::heapArray<byte>(CHUNK_SIZE)),
	start(start), end(end)
{
	KJ_REQUIRE(end >= start);
}

Promise<void> TransmissionProcess::run() {
	reader -> skip(start);
	
	auto request = receiver.beginRequest();
	request.setNumBytes(end - start);
	return request.send().ignoreResult().then([this]() { return transmit(start); });
}

Promise<void> TransmissionProcess::transmit(size_t chunkStart) {			
	// Check if we are done transmitting
	if(chunkStart >= end)
		return receiver.doneRequest().send().ignoreResult();
	
	// Do a transmission
	auto request = receiver.receiveRequest();
	auto buffer = request.initData(kj::min(CHUNK_SIZE, end - chunkStart));
	
	return reader -> tryReadAsync(buffer.begin(), buffer.size(), buffer.size(), getActiveThread().worker())
	.then([this, chunkStart, request = mv(request), buffer](size_t nBytesRead) mutable {
		KJ_REQUIRE(nBytesRead >= buffer.size(), "Insufficient bytes read");
		return request.send().then([this, chunkEnd = chunkStart + buffer.size()]() { return transmit(chunkEnd); });
	});
	// reader -> read(buffer.begin(), buffer.size());
	
	/*if(chunkStart == 0) {
		uint32_t* prefix = reinterpret_cast<uint32_t*>(buffer.begin());
		uint32_t nSegments = prefix[0] + 1;
		
		kj::ArrayPtr<uint32_t> segmentSizes(prefix + 1, nSegments);
		
		size_t expected = nSegments / 2 + 1;
		for(auto s : segmentSizes)
			expected += s;
		
		KJ_DBG("Transmitting from ODB", end, nSegments, segmentSizes, expected, end / 8);
	}*/
	
}

OneOf<decltype(nullptr), Capability::Client, kj::Exception> createInterface(ObjectDBEntry& entry, kj::Badge<ObjectDBHook> badge) {
	entry.trySync();
	
	if(!entry.hasPreserved())
		return nullptr;
	
	auto reader = entry.loadPreserved(true); // Do a shallow load to check contents
	switch(reader -> which()) {
		case ObjectInfo::UNRESOLVED:
			return nullptr;
		case ObjectInfo::NULL_VALUE:
			return Capability::Client(nullptr);
		case ObjectInfo::EXCEPTION: {
			return fromProto(reader -> getException());
		}	
		case ObjectInfo::LINK: {
			return entry.loadPreserved(false) -> getLink();
		}
		case ObjectInfo::DATA_REF: {
			auto refInfo = reader -> getDataRef();
			
			if(refInfo.getDownloadStatus().isDownloading())
				return nullptr;
			
			// Load snapshot
			/*auto& snapshot = entry.getSnapshot();
			auto blobId = snapshot.getBlob.bind(entry.id);
			blobId.step();
			auto blob = snapshot.blobStore -> get(blobId[0]);*/
			
			return Capability::Client(kj::heap<DataRefInterface>(entry.addRef(), badge));
		}
		case ObjectInfo::FOLDER:
			return Capability::Client(kj::heap<FolderInterface>(entry.addRef(), badge));
		case ObjectInfo::FILE:
			return Capability::Client(kj::heap<FileInterface>(entry.addRef(), badge));
	}
	
	KJ_FAIL_REQUIRE("Unknown object type");
}

}}

namespace fsc {
	
Warehouse::Client openWarehouse(db::Connection& conn, bool readOnly, kj::StringPtr tablePrefix) {
	// Make sure we use a forkable connection
	conn.fork(true);
	
	// We do not want the connection to auto-checkpoint (this will cause a large number of fsync calls)
	conn.exec("PRAGMA wal_autocheckpoint = 0");
	
	return kj::refcounted<ObjectDB>(conn, tablePrefix, readOnly);
}

}
