#include "odb.h"
#include "data.h"

#include <capnp/rpc.capnp.h>

#include <set>

using kj::str;

using namespace capnp;

namespace fsc { namespace {
	
template<typename T>
auto withODBBackoff(T func) {
	return withBackoff(10 * kj::MILLISECONDS, 5 * kj::MINUTES, 2, mv(func));
}

// ============================= SQL Database Structure =================================

struct ObjectDBBase {
	ObjectDBBase(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly);
	
	Statement createObject;
	Statement setInfo;
	Statement incRefcount;
	Statement decRefcount;
	Statement deleteObject;
	
	Statement insertRef;
	Statement clearOutgoingRefs;
	
	Statement getInfo;
	Statement listOutgoingRefs;
	Statement getRefcount;
	
	Statement getBlob;
	Statement setBlob;
	
	Statement readNewestId;
	
	kj::String tablePrefix;
	Own<db::Connection> conn;
	bool readOnly;
	Own<BlobStore> blobStore;
	
	Maybe<int64_t> getNewestId();
};

ObjectDBBase::ObjectDBBase(db::Connection& paramConn, kj::StringPtr paramTablePrefix, bool paramReadOnly) :
	tablePrefix(kj::heapString(paramTablePrefix)),
	conn(paramConn -> addRef),
	readOnly(paramReadOnl),
	blobStore(createBlobStore(*conn, tablePrefix, readOnly))
{		
	auto objectsTable = str(tablePrefix, "_objects");
	auto refsTable = str(tablePrefix, "_object_refs");
	auto blobsTable = str(tablePrefix, "_blobs");
	
	if(!readOnly) {
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", objectsTable, " ("
			  "id INTEGER PRIMARY KEY AUTOINCREMENT,"
			  "info BLOB DEFAULT x'00 00 00 00  00 00 00 01  00 00 00 00  00 00 00 00',"
			  "refcount INTEGER DEFAULT 0,"
			  "blobId INTEGER DEFAULT NULL REFERENCES ", blobsTable, "(id)"
			")"
		));
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", refsTable, " ("
			  "parent INTEGER REFERENCES ", objectsTable, "(id) ON DELETE CASCADE ON UPDATE CASCADE,"
			  "slot INTEGER,"
			  "child INTEGER REFERENCES ", objectsTable, "(id)"
			")"
		));
		conn -> exec(str(
			"CREATE UNIQUE INDEX IF NOT EXISTS ", tablePrefix, "_index_refs_by_parent ON ", refsTable, "(parent, slot)"
		));
		
		createObject = conn -> prepare(str("INSERT INTO ", objectsTable, " DEFAULT VALUES"));
		setInfo = conn -> prepare(str("UPDATE ", objectsTable, " SET info = ?2 WHERE id = ?1"));
		incRefcount = conn -> prepare(str("UPDATE ", objectsTable, " SET refcount = refcount + 1 WHERE id = ?"));
		decRefcount = conn -> prepare(str("UPDATE ", objectsTable, " SET refcount = refcount - 1 WHERE id = ?"));
		deleteObject = conn -> prepare(str("DELETE FROM ", objectsTable, " WHERE id = ?"));
		
		setBlob = conn -> prepare(str("UPDATE ", objectsTable, " SET blobId = ?2 WHERE id = ?1"));
		
		insertRef = conn -> prepare(str("INSERT INTO ", refsTable, " (parent, slot, child) VALUES (?, ?, ?)"));
		clearOutgoingRefs = conn -> prepare(str("DELETE FROM ", refsTable, " WHERE parent = ?"));
	}
	
	getInfo = conn -> prepare(str("SELECT info FROM ", objectsTable, " WHERE id = ?"));
	getRefcount = conn -> prepare(str("SELECT refcount FROM ", objectsTable, " WHERE id = ?"));
	getBlob = conn -> prepare(str("SELECT blobId FROM ", objectsTable, " WHERE id = ?"));
	
	listOutgoingRefs = conn -> prepare(str("SELECT child FROM ", refsTable, " WHERE parent = ? ORDER BY slot"));
	
	readNewestId = conn -> prepare(str("SELECT seq FROM sqlite_sequence WHERE name = ?"));
}

int64_t ObjectDBBase::getNewestId() {
	auto& q = readNewestId.bind(kj::str(tablePrefix, "_objects"));
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
	void trySync();
	
	/**
	  * Register a promise that fires when the preserved view moves to a new snapshot
	  * and that throws when the object gets deleted before a new snapshot sees it.
	  */
	Promise<void> whenUpdated();
	
	//! Load the last recently seen representation (read-only) of the object.
	Own<ObjectInfo::Reader> loadPreserved();
	
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

private:
	Own<ObjectInfo::Reader> doLoad(Maybe<ObjectDBSnapshot&> snapshot);
	
	bool isLive() { return liveLink.isLinked(); }
	
	Own<ObjectDB> parent;
	int64_t id;
	Maybe<Own<ObjectDBSnapshot>> snapshot;
		
	Maybe<Own<kj::Fulfiller<void>>> updateFulfiller;
	ForkedPromise<void> updatePromise;
	
	kj::ListLink<ObjectDBEntry> liveLink;
	friend class ObjectDB;
	friend class ObjectEntryData;
};

/** Read-modify-write accessor for database objects
 *
 * This class maintains a builder that can be modified to change the state of the object.
 * Upon destruction (or release(true) ), the modified representation will be written back
 * into the database. The accessor implicitly maintains a database transaction.
 */
struct ObjectDBEntry::Accessor : public ObjectInfo::Builder {
	Accessor(ObjectDBEntry& e) : transaction(e.parent -> conn), target(e.addRef()) { load(); }
	~Accessor() { if(transaction.active() && !ud.isUnwinding()) save(); }
	
	/**
	 * Release the built-in transaction, optionally with (true) or without (false) writing the
	 * object representation back to the database.
	 */
	void release(bool save) { if(save) { save(); } transaction.commit(); }
	
private:
	void load();
	void save();
	
	kj::UnwindDetector ud;
	
	db::Transaction transaction;
	Own<ObjectDBEntry> target;
	
	MallocMessageBuilder messageBuilder;
	BuilderCapabilityTable capTable;
};

/** Object database
 *
 * This class maintains the main connection to the database, as well as a frozen
 * snapshot view of the database, that can be repeatedly updated.
 */
struct ObjectDB : public ObjectDBBase, kj::Refcounted {
	ObjectDB(db::Connection& conn, kj::StringPtr tablePrefix, bool readOnly);
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
	Own<ObjectDBEntry> open(int64_t id, Maybe<ObjectDBSnapshot&> snapshot);
	
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
	OneOf<Own<ObjectDBEntry>, Capability::Client, nullptr> unwrap(Capability::Client);
	
	//! Wraps a DB object (or null value) as a capability client.
	Capability::Client wrap(Maybe<Own<ObjectDBEntry>> e);
	
	//! Unwraps an object KNOWN to be pointing to a database entry
	Own<ObjectDBEntry> unwrapValid(Capability::Client c) { return mv(unwrap(c).get<Own<ObjectDBEntry>>()); }
	
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

private:
	Maybe<Own<ObjectDBSnapshot>> currentSnapshot;
	
	kj::List<ObjectDBEntry, &ObjectDBEntry::liveLink> liveEntries;
	friend class ObjectDBEntry;
	
	kj::TaskSet importTasks;
	friend class ImportContext;
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

	kj::Maybe<ClientHook&> getResolved() override {	return *inner; }
	kj::Maybe<kj::Promise<kj::Own<ClientHook>>> whenMoreResolved() override {
		return inner -> whenMoreResolved();
	}

	virtual kj::Own<ClientHook> addRef() override { return kj::addRef(*this); }
	const void* getBrand() override { return &BRAND; }
	kj::Maybe<int> getFd() override { return nullptr; }
	
	// Implementation method
	ObjectHook(Own<ObjectDBEntry> objectIn);
	
	//! Checks whether the backing object has resolved / thrown.
	kj::Maybe<Capability::Client> checkResolved();
	
	//! Task to periodically check the backing object and resolve eventually.
	Promise<ClientHook> resolveTask();

	// Member
	inline static const uint BRAND = 0;
	Own<ObjectDBEntry> entry;
	Own<ClientHook> inner;
};

//! Glue function that creates the access interface for entries.
Maybe<Capability::Client> createInterface(ObjectDBEntry& e);

// class ObjectDBBase

// class ObjectDB
	
ObjectDBSnapshot& ObjectDB::getCurrentSnapshot() {
	KJ_IF_MAYBE(p, currentSnapshot) {
		return **p;
	}
	
	// We need to open a fresh snapshot
	auto newSnapshot = kj::refcounted<ObjectDBSnapshot>(*this);
	currentSnapshot = newSnapshot -> addRef();
	
	return newSnapshot;
}

void ObjectDB::changed() {
	currentSnapshot = nullptr;
}

void ObjectDB::syncAll() {
	for(auto& e : liveEntries) {
		e.trySync();
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
		return kj::refcounted<ObjectDBHook>(mv(*pEntry));
	}
	return nullptr;
}

OneOf<Own<ObjectDBEntry>, Capability::Client, nullptr> ObjectDB::unwrap(Capability::Client clt) {
	// Resolve to innermost hook
	// If any of the hooks encountered is an object hook, use that
	Own<capnp::ClientHook> hook = capnp::ClientHook::from(kj::cp(object));
	
	ClientHook* inner = hook.get();
	
	// Find INNERmost hook that matches
	Own<ObjectDBHook> matchingHook = nullptr;
	while(true) {
		if(inner -> getBrand() == &ObjectHook::BRAND) {
			auto asObjectHook = static_cast<ObjectHook*>(inner);
			
			if(asObjectHook -> entry -> parent.get() == this) {
				matchingHook = asObjectHook -> addRef();
			}
		}
		
		KJ_IF_MAYBE(pHook, inner -> getResolved()) {
			inner = pHook;
		} else {
			break;
		}
	}
	
	if(matchingHook != nullptr) {
		return matchingHook -> entry -> addRef();
	}
	
	// ... otherwise check whether it is a null cap ... 
	if(inner -> isNull()) {
		return nullptr;
	}
	
	// ... turns out we can't unwrap this.
	return mv(object);
}

// class ObjectDBEntry

ObjectDBEntry::ObjectDBEntry(ObjectDB& parent, int64_t id, Maybe<ObjectDBSnapshot&> paramSnapshot) :
	parent(parent.addRef()), id(id)
{
	KJ_IF_MAYBE(pSnap, paramSnapshot) {
		snapshot = pSnap -> addRef();
	}
	
	parent.liveObjects.add(*this);
	trySync();
}

ObjectDBEntry::~ObjectDBEntry() {
	if(isLive()) {
		parent -> liveObjects.remove(*this);
	}
}

void ObjectDBEntry::trySync() {
	if(!isLive())
		return;
	
	ObjectDBSnapshot& dbSnapshot = parent -> getCurrentSnapshot();
	
	KJ_IF_MAYBE(pMySnap, snapshot) {
		// We are already on the current snapshot
		if(pMySnap -> get() == &dbSnapshot)
			return;
	}
	
	// Check if the object exists
	auto& getRefcount = dbSnapshot.getRefcount;
	getRefcount.bind(id);
	
	if(!getRefcount.step()) {
		// Either object is dead, or not yet in the database
		KJ_IF_MAYBE(pId, dbSnapshot.getNewestId()) {
			if(*pId >= id) {
				// Object is dead
				KJ_IF_MAYBE(ppFulfiller, updateFulfiller) {
					(**ppFulfiller).reject(objectDeleted());
				}
				
				parent -> liveObjects.remove(*this);
			}
		}
		
		return;
	}
	
	// Update to new snapshot
	snapshot = dbSnapshot.addRef();
	
	// Notify waiting updates
	KJ_IF_MAYBE(ppFulfiller, updateFulfiller) {
		(**ppFulfiller).fulfill();
		
		auto paf = kj::newPromiseFulfillerPair<void>();
		updateFulfiller = mv(paf.fulfiller);
		updatePromise = paf.promise.fork();
	}
}

ObjectDBSnapshot& ObjectDBEntry::getSnapshot() {
	KJ_IF_MAYBE(pMySnap, snapshot) {
		return &pMySnap;
	}
	
	KJ_FAIL_REQUIRE("Snapshot requested on object without snapshot");
}

Promise<void> ObjectDBEntry::whenUpdated() {
	if(!isLive()) {
		return objectDeleted();
	}
	
	if(updateFulfiller == nullptr) {
		auto paf = kj::newPromiseFulfillerPair<void>();
		updateFulfiller = mv(paf.fulfiller);
		updatePromise = paf.promise.fork();
	}
	
	return updatePromise.addBranch();
}

Own<ObjectInfo::Reader> ObjectDBEntry::loadPreserved() {	
	KJ_IF_MAYBE(pSnap, snapshot) {
		return doLoad(**pSnap);
	}
	KJ_FAIL_REQUIRE("loadPreserved() has no snapshot to load from. Ensure that hasPreserved() is true, e.g. by waiting for whenUpdated()", id);
}

Own<ObjectInfo::Builder> ObjectDBEntry::load() {
	auto transaction = kj::heap<db::Transaction>(parent -> conn);
	auto reader = doLoad(nullptr);
	
	auto messageBuilder = kj::heap<MallocMessageBuilder>();
	messageBuilder -> setRoot(*reader);
	
	auto builder = kj::heap<ObjectInfo::Builder>(builder.getRoot<ObjectInfo>());
	return builder.attach(mv(messageBuilder), mv(transaction));
}

// class ObjectDB

void ObjectDB::deleteIfOrphan(int64_t id) {
	db::Transaction(*conn);
	
	KJ_REQUIRE(!readOnly);
	
	// Make sure that the object is not in use
	auto& rc = getRefcount.bind(id);
	KJ_REQUIRE(rc.step(), "Internal error, refcount not found");
	
	if(rc[0].asInt64() > 0)
		return;
	
	auto& blobId = getBlobId().bind(id);
	blobId.step();
	
	if(!blobId[0].isNull()) {
		blobStore -> get(blobId[0].asInt64()) -> decRef();
	}
		
	// Scan outgoing refs
	std::set<int64_t> idsToCheck;
	{
		auto& outgoingRefs = listOutgoingRefs.query(id);
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
	
	// Check if we have capabilities without refs
	for(int64_t id : idsToCheck) {
		deleteIfOrphan(id);
	}	
}

Own<ObjectDBEntry> ObjectDB::create() {
	int64_t id = createObject.insert();
	auto result = open(id);
	
	KJ_DBG(ObjectInfo::Reader(*result -> open()));
	return result;
}

Own<ObjectInfo::Reader> ObjectDBEntry::doLoad(Maybe<ObjectDBSnapshot&> snapshotToUse) {
	auto getTargetBase = [&]() -> ObjectDBBase& {
		KJ_IF_MAYBE(pSnap, snapshotToUse) {
			return *pSnap:
		}
		
		return *parent;
	}
	ObjectDBBase& targetBase = getTargetBase();
	
	// Start transaction
	db::Transaction(targetBase.conn);
	
	auto& q = parent.getInfo.bind(id);
	KJ_REQUIRE(q.step(), "Object not present in database");
	
	auto flatInfo = kj::heapArray<const byte>(q[0].asBlob());
	auto heapBuffer = bytesToWords(mv(flatInfo));
	
	kj::Vector<Maybe<Own<ClientHook>>> rawCapTable;
	
	auto& refs = targetBase.listOutgoingRefs.bind(id);
	while(refs.step()) {		
		if(refs[0].isNull()) {
			rawCapTable.add(nullptr);
			continue;
		}
		
		auto dbObject = parent -> open(refs[0], snapshotToUse);
		rawCapTable.add(ClientHook::from(parent -> wrap(mv(dbObject))));
	}
	
	Own<ReaderCapabilityTable> capTable(rawCapTable.releaseAsArray());
	
	auto messageReader = kj::heap<FlatArrayMessageReader>(heapBuffer).attach(mv(heapBuffer));
	auto root = capTable -> imbue(reader -> getRoot<ObjectInfo>());
	
	return kj::heap<ObjectInfo::Reader>(root).attach(mv(messageReader), mv(capTable));
}

Own<ObjectDBEntry::Accessor> ObjectDBEntry::open() {
	return kj::heap<Accessor>(*this);
}

// class ObjectDBEntry::Accessor

void ObjectDBEntry::Accessor::load() {
	KJ_REQUIRE(!target -> parent -> readOnly);
	auto inputReader = target -> doLoad(nullptr);
	
	auto outputPtr = capTable.imbue(messageBuilder.initRoot<AnyPointer>());
	outputPtr.setAs(*inputReader);
	
	*this = outputPtr.getAs<ObjectInfo>();
}

void ObjectDBEntry::Accessor::save(ObjectInfo::Reader input) {
	KJ_REQUIRE(!parent -> readOnly);
	
	// Decrease refcount of all previous outgoing references
	std::set<int64_t> idsToCheck;
	{
		auto& outgoingRefs = parent -> listOutgoingRefs.bind(id);
		while(outgoingRefs.step()) {
			// Skip NULL
			if(outgoingRefs[0].isNull()) {
				continue;
			}
			
			int64_t target = outgoingRefs[0];;
			parent -> decRefcount(target);
			idsToCheck.insert(target);
		}
	}
	
	// Clear existing references
	parent -> clearOutgoingRefs(id);
	
	// Serialize the message into the database
	kj::Array<byte> flatInfo = wordsToBytes(messageToFlatArray(messageBuilder));
	parent -> setInfo(id, flatInfo.asPtr());
	
	// Iterate through the captured capabilities and store the links in the appropriate DB table
	kj::ArrayPtr<kj::Maybe<kj::Own<ClientHook>>> capTableData = capTable.getTable();
	for(auto i : kj::indices(capTableData)) {
		KJ_IF_MAYBE(pHook, capTableData[i]) {
			Capability::Client client((*pHook) -> addRef());
			auto unwrapped = parent -> unwrap(client);
			
			KJ_REQUIRE(!unwrapped.is<Capability::Client>(), "Only immediate DB references and null capabilities may be used inside DBObject.");
			if(unwrapped.is<Own<DBObject>>()) {
				auto& target = unwrapped.get<Own<DBObject>>();
				
				auto rowid = parent -> insertRef.insert(id, (int64_t) i, target -> id);
				parent -> incRefcount(target -> id);
				
				continue;
			}
		}
		
		parent -> insertRef(id, (int64_t) i, nullptr);
	}
	
	// Check if we have capabilities without incoming refs
	for(int64_t id : idsToCheck) {
		parent -> deleteIfOrphan(id);
	}
	
	parent -> changed();
}

// class ObjectDBHook

ObjectDBHook::ObjectDBHook(Own<ObjectDBEntry> paramEntry) :
	entry(mv(paramEntry))
{	
	inner = capnp::newLocalPromiseClient(resolveTask());
}

Promise<ClientHook> ObjectDBHook::resolveTask() {
	return withODBBackoff([this]() -> Promise<ClientHook>  {
		KJ_IF_MAYBE(pClient, checkResolved()) {
			return ClientHook::from(mv(*pClient));
		}
		
		return entry -> whenUpdated()
		.then([this]() {
			return resolveTask();
		});
	});
}

Maybe<Capability::Client> checkResolved() {
	return createInterface(*entry);
}

// ======================= Import interfaces ===================

struct ImportContext;
struct DatarefDownloadProcess;

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
	Maybe<Own<ObjectDBEntry>> import(Capability::Client target);
	
private:
	//! Kicks off import tasks, called by destructor.
	void scheduleImports();
	Promise<void> importTask(Own<ObjectDBEntry>, Capability::Client target);
	
	Own<ObjectDB> parent;
	Transaction t;
	kj::UnwindDetector ud;
	
	struct PendingImport {
		Capability::Client src;
		Own<ObjectDBEntry> dst;
	};
	std::list<PendingImport> pendingImports;
};

struct DatarefDownloadProcess : public internal::DownloadTask<Own<ObjectDBEntry>> {
	DatarefDownloadProcess(Own<ObjectDBEntry> dst, DataRef<AnyPointer>::Client src) :
		DownloadTask(src, Context()),
		dst(mv(dst))
	{}
	
	Promise<Maybe<Own<ObjectDBEntry>>> unwrap() override;
	Promise<Maybe<Own<DBObject>>> useCached() override;
	Promise<void> receiveData(kj::ArrayPtr<const kj::byte> data) override;
	Promise<void> finishDownload() override;
	Promise<Own<DBObject>> buildResult() override;
	
	ObjectDB& db;
	int64_t id;
	
	Own<BlobBuilder> builder;
};

// class ImportContext

ImportContext::ImportContext(ObjectDB& parent) :
	parent(parent.addRef())
{
	KJ_REQUIRE(!parent.conn -> inTransaction(), "ImportContext must be created outside any transactions");
}

ImportContext::~ImportContext() {
	if(!ud.isUnwinding()) {
		t.commit();
		scheduleImports();
	}
}

Maybe<Own<DBObjectEntry>> import(Capability::Client target) {
	auto unwrapped = parent -> unwrap(mv(target));
	
	if(unwrapped.is<decltype(nullptr)>())
		return nullptr;
	
	if(unwrapped.is<Own<ObjectDBEntry>>())
		return mv(unrapped.get<Own<ObjectDBEntry>>());
	
	auto newEntry = parent.create();
	
	PendingImport import { newEntry -> addRef(), target };
	pendingImports.push_back(mv(import));
	
	return newEntry;
}

void ImportContext::scheduleImports() {
	for(auto& import : pendingImports) {
		parent -> importTasks().add(importTask(mv(import.dst), mv(import.src)));
	}
}

void ImportContext::importTask(Own<ObjectDBEntry> dst, Capability::Client src) {
	ObjectDB& myParent = *parent;
	int64_t id = dst -> id;
	
	// Attempt to download DataRef
	auto dlp = kj::refcounted<DatarefDownloadProcess>(parent, dst -> id, src);
	return dlp.result()
	
	// Check what type of exception it is
	.catch_([](kj::Exception&& e) -> Promise<Own<ObjectDBEntry>> {
		if(e.getType() == kj::EXCEPTION::UNIMPLEMENTED)
			return KJ_EXCEPTION(FAILED, "The only foreign objects allowed in the database are DataRefs");
		
		return mv(e);
	})
	
	// If it fails, store failure in database
	.catch_([&myParent, id](kj::Exception&& e) mutable -> Promise<Own<ObjectDBEntry>> {
		return withODBBackoff([&myParent, id]() mutable {
			myParent.open(id)
				-> open()
				-> setException(
					toProto(e)
				)
			);
			
			return mv(e);
		});
	});
}

// class DatarefDownloadProcess

DatarefDownloadProcess::DatarefDownloadProcess(ObjectDB& db, int64_t id, DataRef<AnyPointer>::Client src) :
	DownloadTask(src, Context()),
	db(db), id(id)
{}

Promise<Maybe<Own<ObjectDBEntry>>> DatarefDownloadProcess::unwrap() {
	return withODBBackoff([this]() mutable -> Maybe<Own<DBObject>> {
		auto unwrapResult = db.unwrap(src);
		
		if(unwrapResult.is<Own<DBObject>>()) {
			auto acc = db.open(id) -> open();
			acc -> setLink(src);
			acc -> release(true);
			
			return mv(dst);
		}
		
		return nullptr;
	});
}

Promise<Maybe<Own<DBObject>>> DatarefDownloadProcess::useCached() {
	return withODBBackoff([this]() mutable -> Maybe<Own<DBObject>> {
		ImportContext ic(*parent);
		auto acc = db.open(id) -> open();
		
		// Initialize target to DataRef
		auto ref = acc -> initDataRef();
		ref.setMetadata(metadata);
		ref.getMetadata().setDataHash(nullptr);
		
		auto capTableOut = ref.initCapTable(capTable.size());
		for(auto i : kj::indices(capTable))
			capTableOut.set(i, ic.import(capTable[i]));
		
		// Check if we have hash in blob store
		KJ_IF_MAYBE(pBlob, dst -> parent -> blobStore -> find(metadata.getDataHash())) {
			ref.getDownloadStatus().setFinished();
			pBlob -> incRef();
			return mv(dst);
		}
		
		// Allocate blob builder
		builder = dst -> parent -> blobStore -> create();
		
		// Set blob id under construction so that the blob gets deleted with
		// the parent object.
		dst -> parent -> setBlob(dst -> id, builder -> getId());
		
		return nullptr;
	});
}
	
Promise<void> DatarefDownloadProcess::receiveData(kj::ArrayPtr<const kj::byte> data) {
	return withODBBackoff([this, data]() mutable {
		KJ_REQUIRE(builder.get() != nullptr);
		
		// BlobBuilder::write() does not go inside a transaction
		builder -> write(data);
	});
}

Promise<void> DatarefDownloadProcess::finishDownload() {
	return withODBBackoff([this]() mutable {
		KJ_REQUIRE(builder.get() != nullptr);
		
		// This goes outside the transaction
		builder -> prepareFinish();
		
		// This goes inside the transaction
		db::Transaction t(db.conn);
		auto finishedBlob = builder -> finish();
		db.setBlob(id, finishedBlob -> getId());
	});
}

Promise<Own<DBObject>> DatarefDownloadProcess::buildResult() {
	return mv(dst);
}	

// ======================= Access interfaces ===================

struct FolderInterface;
struct DataRefInterface;
struct TransmissionProcess;

//! Interface class for folder objects
struct FolderInterface : public ::fsc::odb::Folder::Server {
	// Interface
	Promise<void> ls(LsContext ctx) override;
	Promise<void> getAll(GetAllContext ctx) override;
	Promise<void> getEntry(GetEntryContext ctx) override;
	Promise<void> putEntry(PutEntryContext ctx) override;
	Promise<void> rm(RmContext ctx) override;
	Promise<void> mkdir(MkdirContext ctx) override;

	// Implementation
	FolderInterface(Own<ObjectDBEntry>&& object, kj::Badge<ObjectDBHook>) :
		object(mv(object))
	{}
	
	//! Sets a folder entry. Passing nullptr as 3rd arg deletes it instead.
	void setEntry(ObjectInfo::Folder::Builder folder, kj::StringPtr name, Maybe<Own<ObjectDBEntry>> entry);
	
	//! Opens a folder for modification. Optionally creates the folder (and all parents) if not present
	Own<ObjectDBEntry::Accessor> getFolderForModification(kj::PathPtr path, bool create = false);
	
	//! Reads an object relative to this one. Uses the preserved snapshot of this folder.
	Own<ObjectInfo::Reader> readObject(kj::PathPtr path);
	
	// Members
	Own<ObjectDBEntry> object;
};

struct DataRefInterface : public DataRef::Server {
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
	
	Own<kj::InputStream> reader;
	
	DataRef<capnp::AnyPointer>::Receiver::Client receiver;
	size_t start;
	size_t end;
	
	Array<byte> buffer;
	
	TransmissionProcess(Own<kj::InputStream>&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end);
	
	Promise<void> run();
	Promise<void> transmit(size_t chunkStart);
};

// class FolderInterface

Promise<void> FolderInterface::ls(LsContext ctx) {
	auto data = readObject(kj::Path::parse(ctx.getParams().getName()));
	
	KJ_REQUIRE(data -> isFolder(), "Trying to list contents of non-folder object");
	
	auto in = data -> getFolder().getEntries();
	auto out = ctx.getResults().initEntries(in.size());
	for(auto i : kj::indices(in)) {
		out[i] = in[i].getName();
	}
	
	return READY_NOW;
}

Promise<void> FolderInterface::getAll(GetAllContext ctx) {
	auto data = readObject(kj::Path::parse(ctx.getParams().getName()));
	
	KJ_REQUIRE(data -> isFolder(), "Trying to list contents of non-folder object");
	
	auto in = data -> getFolder().getEntries();
	auto out = ctx.getResults().setEntries(in);
	
	return READY_NOW;
}

Promise<void> FolderInterface::getEntry(GetEntryContext ctx) {
	auto path = kj::Path::parse(ctx.getParams().getName());
	
	auto data = readObject(path.parent());
	KJ_REQUIRE(data -> isFolder(), "Trying to access entry of non-folder object");
	
	auto& filename = path.basename()[0];
	for(auto e : data -> getFolder().getEntries()) {
		if(e.getName().asString() == name) {
			ctx.setResults(e);
			return READY_NOW;
		}
	}
	
	return KJ_EXCEPTION(FAILED, "Entry not found in folder", filename);
}

Promise<void> FolderInterface::putEntry(PutEntryContext ctx) {
	return withODBBackoff([this, ctx]() mutable -> Promise<void> {
		kj::Path path = kj::Path::parse(ctx.getParams().getName());
		ImportContext ic;
		
		auto acc = getFolderForModification(path, true);
		
		auto& filename = path.basename()[0];
		KJ_IF_MAYBE(pEntry, ic.import(ctx.getParams().getValue())) {
			setEntry(*acc, filename, mv(*pEntry));
		} else {
			setEntry(*acc, filename, nullptr);
		}
		
		acc -> release(true);
	});
}

Promise<void> FolderInterface::rm(RmContext ctx) {
	return withODBBackoff([this, ctx]() mutable -> Promise<void> {
		kj::Path path = kj::Path::parse(ctx.getParams().getName());
		
		auto acc = getFolderForModification(path.parent(), false);
		
		auto& filename = path.basename()[0];
		setEntry(*acc, filename, nullptr);
		
		acc -> release(true);
	});
}

Promise<void> FolderInterface::mkdir(MkdirContext ctx) {
	return withODBBackoff([this, ctx] () mutable -> Promise<void> {
		kj::Path path = kj::Path::parse(ctx.getParams().getName());
		
		auto acc = getFolderForModification(path.parent(), true);
		
		auto& filename = path.basename()[0];
		
		auto newObject = object -> parent -> create();
		newObject -> open() -> initFolder();
		
		setEntry(*acc, filename, newObject -> addRef());
		acc -> release(true);
	});
}

FolderInterface::FolderInterface(Own<ObjectDBEntry>&& object, kj::Badge<ObjectDBHook>) :
	object(mv(object))
{}

void FolderInterface::setEntry(ObjectInfo::Folder::Builder folder, kj::StringPtr name, Maybe<Own<ObjectDBEntry>> entry) {
	// Try to locate existing entry with correct name
	Maybe<size_t> maybeIdx;
	
	auto entries = folder.getEntries();
	for(auto i : kj::indices(entries)) {
		if(entries[i].getName() == name) {
			maybeIdx = i;
			break;
		}
	}
	
	KJ_IF_MAYBE(pIdx, maybeIdx) {
		size_t idx = *pIdx;
		
		KJ_IF_MAYBE(pVal, entry) {
			entries[i].setVal(object -> parent -> wrap(mv(*pVal)));
		} else {
			auto orphanage = Orphanage::getForMessageContaining(folder);
			auto newList = orphanage.newOrphan<List<FolderEntry>>(entries.size() - 1);
			for(auto i : kj::range(0, idx)) {
				newList.get().setWithCaveats(i, entries[i]);
			}
			for(auto i : kj::range(idx + 1, entries.size())) {
				newList.get().setWithCaveats(i - 1, entries[i]);
			}
			folder.adoptEntries(mv(newList));
		}
	} else {
		KJ_IF_MAYBE(pVal, entry) {
			auto orphanage = Orphanage::getForMessageContaining(folder);
			
			// Extend entries by 1
			auto disowned = folder.disownEntries();
			disowned.truncate(disowned.get().size() + 1);
			folder.adoptEntries(mv(disowned));
			entries = folder.getrEntries();
			
			// Configure last entry
			auto lastEntry = entries[entries.size() - 1];
			lastEntry.setName(name);
			lastEntry.setVal(object -> parent -> wrap(mv(*pVal)));
		}
	}
}

Own<ObjectDBEntry::Accessor> FolderInterface::getFolderForModification(kj::PathPtr path, bool create = false) {
	Own<ObjectDBEntry> current = object -> addRef();
	
	for(const auto& folderName : path) {
		auto acc = current -> open();
		
		KJ_REQUIRE(acc -> isFolder(), "Trying to access child object of non-folder", path, folderName);
		auto entries = acc -> getFolder().getEntries();
		
		// Check if entry exists
		for(auto i : kj::indices(entries)) {
			if(entries[i].getName() == path[0]) {
				current = object -> parent -> unwrapValid(entries[i].getVal());
				acc -> release(false);
				goto entry_found;
			}
		}
		
		// Folder does not exist
		if(create) {
			auto newFolder = object -> parent -> create();
			auto newFolderAcc = newFolder.open();
			newFolderAcc -> initFolder();
			newFolderAcc -> release(true);
			
			setEntry(acc -> getFolder(), folderName, object -> parent -> wrap(newFolder -> addRef()));
			acc -> release(true);
			
			current = mv(newFolder);
		} else {
			KJ_FAIL_REQUIRE("Folder not found", path, folderName);
		}
		
		entry_found:;
	}
	
	return current -> open();
}

Own<ObjectInfo::Reader> FolderInterface::readObject(kj::PathPtr path) {
	KJ_REQUIRE(object -> hasPreserved(), "Internal error: Folder has no preserved state");
	
	Own<ObjectDBEntry> current = object -> addRef();
	for(const auto& folderName : path.parent()) {
		auto reader = current -> getPreserved();
		
		KJ_REQUIRE(acc -> isFolder(), "Trying to access child object of non-folder", path, folderName);
		auto entries = acc -> getFolder().getEntries();
		
		// Check if entry exists
		for(auto i : kj::indices(entries)) {
			if(entries[i].getName() == path[0]) {
				current = object -> parent -> unwrapValid(entries[i].getVal());
				goto entry_found;
			}
		}
		
		KJ_FAIL_REQUIRE("Folder not found", path, folderName);
	}
	
	return current -> getPreserved();
}

// class DataRefInterface

Promise<void> DataRefInterface::metaAndCapTable(MetaAndCapTableContext ctx) override {
	return withODBBackoff([this, ctx]() mutable {
		auto data = object -> getPreserved();
		
		auto results = ctx.initResults();
		results.setMetadata(data -> getMetadata());
		results.setTable(data -> getCapTable());
		
		return READY_NOW;
	});
}

Promise<void> DataRefInterface::rawBytes(RawBytesContext ctx) override {
	return withODBBackoff([this, ctx]() mutable {
		const uint64_t start = ctx.getParams().getStart();
		const uint64_t end = ctx.getParams().getEnd();
		
		auto snapshot = object -> getSnapshot().addRef();
		
		// Query blob id
		auto& blobIdQuery = snapshot -> getBlob.bind(object -> id);
		KJ_REQUIRE(blobIdQuery.step(), "Internal error: Object deleted", id);
		
		// Open blob
		Own<Blob> blob = snapshot -> blobStore.get(blobIdQuery[0]);
		Own<kj::InputStream> reader = blob -> open();
		reader -> skip(start);
		
		// Decompress into output buffer
		auto data = ctx.getResults().initData(end - start);
		reader -> read(data.begin(), data.size());
	});
}

Promise<void> DataRefInterface::transmit(TransmitContext ctx) override {
	return dataReady()
	.then([this, ctx]() mutable {
		return withODBBackoff([this, ctx]() mutable {
			auto params = ctx.getParams();
		
			auto snapshot = object -> getSnapshot().addRef();
			
			auto& blobIdQuery = snapshot -> getBlob.bind(object -> id);
			KJ_REQUIRE(blobIdQuery.step(), "Internal error: Object deleted", id);
			
			// Open blob
			Own<Blob> blob = snapshot -> blobStore.get(blobIdQuery[0]);
			Own<kj::InputStream> reader = blob -> open();
			
			// Create transmission process
			auto transProc = heapHeld<TransmissionProcess>(mv(reader), params.getReceiver(), params.getStart(), params.getEnd());
			return transProc.run().attach(transProc.x());
		});
	});
}

// class TransmissionProcess
	
TransmissionProcess::TransmissionProcess(Own<kj::InputStream>&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end) :
	reader(mv(reader)),
	receiver(mv(receiver)),
	buffer(kj::heapArray<byte>(CHUNK_SIZE)),
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
	
	auto slice = chunkStart + CHUNK_SIZE <= end ? buffer.asPtr() : buffer.slice(0, end - chunkStart);
	reader -> read(slice.begin(), slice.size());
	
	// Do a transmission
	auto request = receiver.receiveRequest();
	
	if(slice.size() % 8 == 0) {
		// Note: This is safe because we keep this object alive until the transmission
		// succeeds or fails
		auto orphanage = capnp::Orphanage::getForMessageContaining((DataRef<capnp::AnyPointer>::Receiver::ReceiveParams::Builder) request);
		auto externalData = orphanage.referenceExternalData(slice);
		request.adoptData(mv(externalData));
	} else {
		request.setData(slice);
	}
	
	return request.send().then([this, chunkEnd = chunkStart + slice.size()]() { return transmit(chunkEnd); });
}

// Implementation helpers

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

Temporary<rpc::Exception> toProto(kj::Exception& e) {
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

Maybe<Capability::Client> createInterface(ObjectDBEntry& entry) {
	if(!entry.hasPreserved())
		return nullptr;
	
	auto reader = entry.loadPreserved();
	switch(reader -> which()) {
		case ObjectInfo::UNRESOLVED:
			return nullptr;
		case ObjectInfo::EXCEPTION: {
			return fromProto(info -> getException());
		}	
		case ObjectInfo::LINK: {
			return reader -> getLink();
		}
		case ObjectInfo::DATA_REF: {
			auto refInfo = reader -> getDataRef();
			
			if(refInfo.getDownloadStatus().isDownloading())
				return nullptr;
			
			return kj::heap<DataRefInterface>(*entry)
		}
		case ObjectInfo::FOLDER:
			return kj::heap<FolderInterface>(*entry);
	}
}

}}

