#include "odb.h"
#include "data.h"

#include <capnp/rpc.capnp.h>

#include <set>

using kj::str;

using namespace capnp;

namespace fsc { namespace odb {
	
BlobStore::BlobStore(sqlite::Connection& conn, kj::StringPtr tablePrefix, bool readOnly) :
	tablePrefix(kj::heapString(tablePrefix)),
	conn(conn.addRef()),
	readOnly(readOnly)
{
	if(!readOnly) {
		conn.exec(str(
			"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_blobs ("
			"  id INTEGER PRIMARY KEY,"
			"  hash BLOB UNIQUE," // SQLite UNIQUE allows multiple NULL values
			"  refcount INTEGER"
			")"
		));
		conn.exec(str(
			"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_chunks ("
			"  id INTEGER REFERENCES ", tablePrefix, "_blobs(id) ON UPDATE CASCADE ON DELETE CASCADE,"
			"  chunkNo INTEGER,"
			"  data BLOB,"
			""
			"  PRIMARY KEY(id, chunkNo)"
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
}

Maybe<Blob> BlobStore::find(kj::ArrayPtr<const byte> hash) {	
	auto q = findBlob.query(hash);
	if(q.step()) {
		return Blob(*this, q[0]);
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
{}

void Blob::incRef() { KJ_REQUIRE(!parent -> readOnly); parent -> incRefcount(id); }
void Blob::decRef() { KJ_REQUIRE(!parent -> readOnly); parent -> decRefcount(id); parent -> deleteIfOrphan(id); }

// ============================== class BlobBuilder =================================

BlobBuilder::BlobBuilder(BlobStore& parent, size_t chunkSize) :
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

BlobBuilder::~BlobBuilder() {
	ud.catchExceptionsIfUnwinding([this]() {
		parent-> deleteIfOrphan(id);
	});
}

void BlobBuilder::flushBuffer() {
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
	KJ_REQUIRE(parent -> conn -> inTransaction(), "Must be inside transaction");
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
	// auto transaction = parent -> conn -> beginTransaction();
	
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
	readStatement(sqlite::Statement(blob.parent -> conn -> prepare(str("SELECT data FROM ", blob.parent -> tablePrefix, "_chunks WHERE id = ? ORDER BY chunkNo")))),
	readQuery(readStatement.query(blob.id))
{}

bool BlobReader::read(kj::ArrayPtr<byte> output) {
	decompressor.setOutput(output);
	
	while(true) {
		if(decompressor.remainingIn() == 0) {
			KJ_REQUIRE(readQuery.step(), "Missing chunks despite expecting more");
			decompressor.setInput(readQuery[0]);
		}
			
		ZLib::State state = decompressor.step();
		
		if(state == ZLib::FINISHED)
			return true;
		
		if(decompressor.remainingOut() == 0)
			return false;
		
		KJ_ASSERT(decompressor.remainingIn() == 0);
	}
}

// ========================= ObjectDB ==============================

namespace {
	template<typename T>
	auto withODBBackoff(T func) {
		return withBackoff(10 * kj::MILLISECONDS, 5 * kj::MINUTES, 2, mv(func));
	}
	
	KJ_NORETURN (void objectDeleted()) {
		kj::throwFatalException(KJ_EXCEPTION(DISCONNECTED, "Object was deleted from database"));
	}
	
	kj::Exception fromProto(rpc::Exception::Reader) { static_assert(false, "Unimplemented"); }
	Temporary<rpc::Exception> toProto(kj::Exception& e) { static_assert(false, "Unimplemented"); }
}

struct ObjectDB::TransmissionProcess {
	constexpr static inline size_t CHUNK_SIZE = 1024 * 1024;
	
	BlobReader reader;
	
	DataRef<capnp::AnyPointer>::Receiver::Client receiver;
	size_t start;
	size_t end;
	
	Array<byte> buffer;
	
	TransmissionProcess(BlobReader&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end) :
		reader(mv(reader)),
		receiver(mv(receiver)),
		buffer(kj::heapArray<byte>(CHUNK_SIZE)),
		start(start), end(end)
	{
		KJ_REQUIRE(end >= start);
	}
	
	Promise<void> run() {
		auto request = receiver.beginRequest();
		request.setNumBytes(end - start);
		return request.send().ignoreResult().then([this]() { return transmit(start); });
	}
	
	Promise<void> transmit(size_t chunkStart) {			
		// Check if we are done transmitting
		if(chunkStart >= end)
			return receiver.doneRequest().send().ignoreResult();
		
		auto slice = chunkStart + CHUNK_SIZE <= end ? buffer.asPtr() : buffer.slice(0, end - chunkStart);
		reader.read(slice);
		KJ_REQUIRE(reader.remainingOut() == 0, "Buffer should be filled completely");
		
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
};

struct ObjectDB::TransmissionReceiver : public DataRef<AnyPointer>::Receiver::Server {
	Own<BlobBuilder> builder;
	Own<DBObject> parent;
	
	TransmissionReceiver(DBObject& obj) :
		parent(obj.addRef())
	{}
	
	Promise<void> begin(BeginContext ctx) {
		return withODBBackoff([this]() {
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> getParent().conn -> beginRootTransaction(true);
		
			builder = kj::heap<BlobBuilder>(parent -> parent -> blobStore);
		});
	}
	
	Promise<void> receive(ReceiveContext ctx) {
		return withODBBackoff([this, ctx]() mutable {
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> getParent().conn -> beginRootTransaction(true);
			builder -> write(ctx.getParams().getData());
		});
	}
	
	Promise<void> done(DoneContext ctx) {
		return withODBBackoff([this]() {
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> getParent().conn -> beginRootTransaction(true);
			
			Blob blob = builder -> finish();
			
			parent -> load();
			parent -> info.getDataRef().getMetadata().setDataHash(blob.hash());
			parent -> save();
			
			blob.incRef();
		});
	}
};

struct ObjectDB::ObjectImpl : public Object::Server {
	Own<DBObject> object;
			
	//! Waits until this object has settled into a usable state
	Promise<void> whenReady() {
		return withODBBackoff([this]() -> Promise<void> {
			object -> load(); 
			auto info = object -> info;
			switch(info.which()) {
				case ObjectInfo::UNRESOLVED:
					return object -> whenUpdated().then([this]() { return whenReady(); });
				case ObjectInfo::EXCEPTION:
					return fromProto(info.getException());
					
				default:
					return READY_NOW;
			}
		});
	}
	
	Promise<void> whenDownloadReady() {
		return whenReady()
		.then([this]() {
			return withODBBackoff([this]() -> Promise<void> {		
				switch(checkRef().getDownloadStatus().which()) {
					case ObjectInfo::DataRef::DownloadStatus::DOWNLOADING:
						return object -> whenUpdated().then([this]() { return whenDownloadReady(); });
					case ObjectInfo::DataRef::DownloadStatus::FINISHED:
						return READY_NOW;
				}
			});
		});
	}
	
	ObjectInfo::Folder::Builder checkFolder() {
		if(object -> info.which() != ObjectInfo::FOLDER)
			KJ_UNIMPLEMENTED("This database object is not a folder");
		return object -> info.getFolder();
	}
	
	ObjectInfo::DataRef::Builder checkRef() {
		if(object -> info.which() != ObjectInfo::DATA_REF)
			KJ_UNIMPLEMENTED("This database object is not a DataRef");
		return object -> info.getDataRef();
	}
	
	Promise<void> getInfo(GetInfoContext ctx) override {
		return whenReady()
		.then([this, ctx]() mutable {
			Object::Type type;
			auto info = object -> info;
			
			switch(info.which()) {
				case ObjectInfo::FOLDER:
					type = Object::Type::FOLDER;
					break;
				case ObjectInfo::DATA_REF:
					type = Object::Type::DATA;
					break;
				default:
					KJ_FAIL_REQUIRE("Internal error");
			}
			
			ctx.getResults().setType(type);
		});
	}
	
	Promise<void> metadata(MetadataContext ctx) override {
		// The metadata table is only ready once the hash is verified
		return whenDownloadReady()
		.then([this, ctx]() mutable {
			ctx.initResults().setMetadata(checkRef().getMetadata());
		});
	}
	
	Promise<void> capTable(CapTableContext ctx) override {
		return whenReady()
		.then([this, ctx]() mutable {
			return withODBBackoff([this, ctx]() mutable {
				auto t = object -> parent -> conn -> beginTransaction();
				
				auto tableIn = checkRef().getCapTable();
				auto tableOut = ctx.getResults().initTable(tableIn.size());
				for(auto i : kj::indices(tableIn))
					tableOut.set(i, tableIn[i]);
			});
		});
	}
	
	Promise<void> rawBytes(RawBytesContext ctx) override {
		return whenDownloadReady()
		.then([this, ctx]() mutable {
			return withODBBackoff([this, ctx]() mutable {
				auto t = object -> parent -> conn -> beginTransaction();
					
				const uint64_t start = ctx.getParams().getStart();
				const uint64_t end = ctx.getParams().getEnd();
				
				auto refInfo = checkRef();
				KJ_REQUIRE(end < refInfo.getMetadata().getDataSize());
				
				auto hash = refInfo.getMetadata().getDataHash();
				KJ_IF_MAYBE(pBlob, object -> parent -> blobStore -> find(hash)) {
					auto buffer = kj::heapArray<byte>(8 * 1024 * 1024);
					auto reader = pBlob -> open();
					KJ_REQUIRE(end >= start);
					
					// Wind forward the stream until we hit the target point
					uint64_t windRemaining = start;
					while(true) {
						if(windRemaining >= buffer.size()) {
							reader.read(buffer);
							windRemaining -= buffer.size();
						} else {
							reader.read(buffer.slice(0, windRemaining));
							break;
						}
					}
					
					auto data = ctx.getResults().initData(end - start);
					reader.read(data);
				} else {
					objectDeleted();
				}
			});
		});
	}
	
	Promise<void> transmit(TransmitContext ctx) override {
		return whenDownloadReady()
		.then([this, ctx]() mutable {
			return withODBBackoff([this, ctx]() mutable {
				auto params = ctx.getParams();
				
				// Since we preparing a long-running op, fork the blob store so that our main connection doesn't get blocked
				// If the database isn't shared, we can use the main connection (though we have to hope the object doesn't get deleted
				auto forkedStore = kj::refcounted<BlobStore>(
					object -> parent -> shared ? *(object -> parent -> forkConnection(true)) : *(object -> parent -> conn),
					object -> parent -> tablePrefix,
					true
				);
				auto forkedTransaction = forkedStore -> conn -> beginTransaction();
				
				// Note: Read operations can still fail					
				KJ_IF_MAYBE(pBlob, forkedStore -> find(checkRef().getMetadata().getDataHash())) {
					auto reader = pBlob -> open();
					auto transProc = heapHeld<TransmissionProcess>(params.getReceiver(), params.getStart(), params.getEnd());
					return transProc -> run().attach(mv(forkedStore), mv(forkedTransaction), transProc.x());
				} else {
					objectDeleted();
				}
			});
		});
	}
	
	Promise<void> ls(LsContext ctx) override {
		return whenReady()
		.then([this, ctx]() mutable {
			auto f = checkFolder();
			auto in = f.getEntries();
			auto out = ctx.getResults().initEntries(in.size());
			for(auto i : kj::indices(in)) {
				out[i] = in[i].getName();
			}
		});
	}
	
	Promise<void> getAll(GetAllContext ctx) override {
		return whenReady()
		.then([this, ctx]() mutable {
			auto f = checkFolder();
			ctx.getResults().setEntries(f.getEntries());
		});
	}
	
	Promise<void> getEntry(GetEntryContext ctx) override {
		return whenReady()
		.then([this, ctx]() mutable {
			auto f = checkFolder();
			auto name = ctx.getParams().getName();

			for(auto e : f.getEntries()) {
				if(e.getName().asString() == name) {
					ctx.setResults(e);
				}
			}
		});
	}
	
	Promise<void> putEntry(PutEntryContext ctx) override {
		return whenReady()
		.then([this, ctx]() mutable {
			auto f = checkFolder();
			
			auto params = ctx.getParams();
			auto name = params.getName();
			
			auto entries = f.getEntries();
			
			FolderEntry::Builder entry = nullptr;
			bool entryPresent = false;
			
			for(auto i : kj::indices(entries)) {
				auto e = entries[i];
				
				if(e.getName().asString() == name) {
					entry = e;
					entryPresent = true;
					break;
				}
			}
			
			if(!entryPresent) {
				auto orphanage = Orphanage::getForMessageContaining(f);
				auto newList = orphanage.newOrphan<List<FolderEntry>>(1);
			
				auto listRefs = kj::heapArray<List<FolderEntry>::Reader>({newList.get(), f.getEntries()});
				f.adoptEntries(orphanage.newOrphanConcat(listRefs.asPtr()));
				
				entry = f.getEntries()[0];
				entry.setName(name);
			}
			
			auto t = object -> parent -> conn -> beginTransaction();
			
			switch(params.which()) {
				case FolderEntry::REF:
					entry.setRef(object -> parent -> download(params.getRef()));
					break;
				
				case FolderEntry::FOLDER:
					entry.setFolder(params.getFolder());
					break;
			}
			
			object -> save();
		});
	}
};
	
//! Marker hook to indicate that a capability is derived from a database object or being exported as such
struct ObjectDB::ObjectHook : public ClientHook, public kj::Refcounted {
	Own<ClientHook> inner;
	static const uint BRAND;
	Own<DBObject> object;
	
	ObjectHook(Own<DBObject> objectIn) :
		inner(ClientHook::from(Capability::Client(kj::heap<ObjectImpl>(*objectIn)))),
		object(mv(objectIn))
	{}

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

	kj::Maybe<ClientHook&> getResolved() override {
		return *inner;
	}

	kj::Maybe<kj::Promise<kj::Own<ClientHook>>> whenMoreResolved() override {
		return nullptr;
	}

	virtual kj::Own<ClientHook> addRef() override { return kj::addRef(*this); }

	const void* getBrand() override { return &BRAND; }

	kj::Maybe<int> getFd() override { return nullptr; }
};

// ========================= class ObjectDB =============================

namespace {
	struct DownloadErrorHandler : public kj::TaskSet::ErrorHandler {
		void taskFailed(kj::Exception&& exception) override {
			KJ_DBG("Warnin: Download task failed (should never happen)", exception);
		}
		
		static DownloadErrorHandler instance;
	};
}

ObjectDB::ObjectDB(kj::StringPtr filename, kj::StringPtr tablePrefix, bool readOnly) :
	filename(str(filename)), tablePrefix(str(tablePrefix)), readOnly(readOnly), downloadTasks(DownloadErrorHandler::instance)
{
	shared = true;
	if(filename == ":memory:" ||filename == "")
		shared = false;
	
	KJ_LOG(WARNING, "Unshared ObjectDB usage is not tested");
	
	conn = openSQLite3(filename);
	
	auto objectsTable = str(tablePrefix, "_objects");
	auto refsTable = str(tablePrefix, "_object_refs");
	
	if(!readOnly) {
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", objectsTable, " ("
			  "id INTEGER PRIMARY KEY,"
			  "info BLOB,"
			  "refcount INTEGER"
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
		incRefcount = conn -> prepare(str("UPDATE ", objectsTable, " SET refcount = refcount + 1 WHERE ID = ?"));
		decRefcount = conn -> prepare(str("UPDATE ", objectsTable, " SET refcount = refcount - 1 WHERE ID = ?"));
		deleteObject = conn -> prepare(str("DELETE FROM ", objectsTable, " WHERE ID = ?"));
		
		insertRef = conn -> prepare(str("INSERT INTO ", refsTable, " (parent, slot, child) VALUES (?, ?, ?"));
		clearOutgoingRefs = conn -> prepare(str("DELETE FROM ", refsTable, " WHERE parent = ?"));
	}
	
	getInfo = conn -> prepare(str("SELECT info FROM ", objectsTable, " WHERE id = ?"));
	getRefcountAndHash = conn -> prepare(str("SELECT refcount, hash FROM ", objectsTable, " WHERE ID = ?"));
	
	listOutgoingRefs = conn -> prepare(str("SELECT child FROM ", refsTable, " WHERE parent = ?"));
}

Object::Client ObjectDB::wrap(Maybe<Own<DBObject>> object) {
	KJ_IF_MAYBE(pObject, object) {
		Capability::Client innerClient(kj::heap<ObjectImpl>(mv(*pObject)));
		
		Own<ClientHook> wrapper = kj::heap<ObjectHook>(ClientHook::from(mv(innerClient)));
		return Object::Client(mv(wrapper));
	}
	
	return nullptr;
}

OneOf<Capability::Client, Own<DBObject>, decltype(nullptr)> ObjectDB::unwrap(Capability::Client object) {
	// Resolve to innermost hook
	// If any of the hooks encountered is an object hook, use that
	Own<capnp::ClientHook> hook = capnp::ClientHook::from(kj::cp(object));
	
	// First check whether this is a DB object in our database ...
	ClientHook* inner = hook.get();
	while(true) {
		if(inner -> getBrand() == &ObjectHook::BRAND) {
			auto asObjectHook = static_cast<ObjectHook*>(inner);
			
			if(asObjectHook -> object -> parent.get() == this)
				return asObjectHook -> object -> addRef();
		}
		
		KJ_IF_MAYBE(pHook, inner -> getResolved()) {
			inner = pHook;
		} else {
			break;
		}
	}
	
	// ... only then (for security reasons) check whether we are downloading it ...
	inner = hook.get();
	while(true) {
		// We are downloading this thing already
		// KJ_IF_MAYBE(pId, activeDownloads.find(inner)) {
		auto pId = activeDownloads.find(inner);
		if(pId != activeDownloads.end()) {
			return kj::refcounted<DBObject>(*this, *pId);
		}
		
		KJ_IF_MAYBE(pHook, inner -> getResolved()) {
			inner = pHook;
		} else {
			break;
		}
	}
	
	// ... otherwise check whether it is a null cap ... 
	if(inner -> isNull()) {
		return nullptr;
	}
	
	// ... turns out we can't unwrap this.
	return mv(object);
}

Object::Client ObjectDB::download(DataRef<AnyPointer>::Client object) {
	auto unwrapped = unwrap(object);
	
	if(unwrapped.is<decltype(nullptr)>()) {
		return nullptr;
	}
	
	if(unwrapped.is<Own<DBObject>>()) {
		return wrap(mv(unwrapped.get<Own<DBObject>>()));
	}
	
	return withODBBackoff([unwrapped = mv(unwrapped), this]() mutable {
		auto asCap = unwrapped.get<Capability::Client>();
		Own<DBObject> dbObject = startDownloadTask(asCap.castAs<DataRef<AnyPointer>>());
		return wrap(mv(dbObject));
	}).attach(addRef());
}

Own<DBObject> ObjectDB::startDownloadTask(DataRef<AnyPointer>::Client object) {	
	// Initialize the object to an unresolved state
	Own<DBObject> dbObject;
	{
		auto transaction = conn -> beginTransaction();
		dbObject = kj::refcounted<DBObject>(*this, createObject.insert(), DBObject::CreationToken());
		dbObject -> info.setUnresolved();
		dbObject -> save();
	}
	
	// Remember that we are downloading this capability into the given address
	// This will prevent starting double-downloads until the hash is actually
	// present in our blob db.
	ClientHook* pHook = ClientHook::from(cp(object)).get();
	activeDownloads.insert({pHook, dbObject -> id});
	
	// Start the download task
	ForkedPromise<void> exportTask = downloadTask(object.castAs<DataRef<AnyPointer>>(), *dbObject)
	
	// If the download fails, store the failure
	.catch_([this, object, dbObject = dbObject -> addRef()](kj::Exception& e) mutable {
		return withODBBackoff([this, e, dbObject = mv(dbObject)]() mutable {
			/*if(e.getType() == kj::Exception::Type::UNIMPLEMENTED)
				dbObject -> info.setUnknownObject();
			else
				dbObject -> info.setException(toProto(e));*/
			dbObject -> info.setException(toProto(e));
			// TODO: Make prettier error message
			
			dbObject -> save();
		});
	})
	// If the failure storage also fails, discard the error, database is probably
	// having larger issues right now anyway
	.catch_([](kj::Exception& e) {})
	
	// Remove exported hook after conclusion
	.then([this, object, id = dbObject -> id, pHook]() {
		activeDownloads.erase(pHook);
		whenResolved.erase(id);
	})
	.attach(dbObject -> addRef())
	.fork();
	
	whenResolved.insert({dbObject -> id, exportTask.addBranch().fork()});
	downloadTasks.add(exportTask.addBranch());
	
	return dbObject;
}

Promise<void> ObjectDB::downloadTask(DataRef<AnyPointer>::Client src, DBObject& dst) {
	using RemoteRef = DataRef<AnyPointer>;
	using MetadataResponse = Response<RemoteRef::MetadataResults>;
	using CapTableResponse = Response<RemoteRef::CapTableResults>;
	
	// Download metadata and capability table
	auto metadataPromise = src.metadataRequest().send().eagerlyEvaluate(nullptr);
	auto capTablePromise = src.capTableRequest().send().eagerlyEvaluate(nullptr);
	
	// Wait for both to be downloaded
	return metadataPromise
	.then([this, src, &dst, capTablePromise = mv(capTablePromise)](MetadataResponse metadataResponse) mutable {
	return capTablePromise.then([this, src, &dst, metadataResponse = mv(metadataResponse)](CapTableResponse capTableResponse) mutable {
	return withODBBackoff([this, src, &dst, metadataResponse = mv(metadataResponse), capTableResponse = mv(capTableResponse)]() mutable -> Promise<void> {
		auto t = conn -> beginTransaction();
		
		auto refData = dst.info.initDataRef();
		
		refData.setMetadata(metadataResponse.getMetadata());
		refData.getMetadata().setDataHash(nullptr);
		
		// Copy cap table over, wrap child objects in their own download processes
		auto capTableIn = capTableResponse.getTable();
		auto capTableOut = refData.initCapTable(capTableIn.size());
		for(auto i : kj::indices(capTableIn)) {
			capTableOut.set(i, download(capTableIn[i].castAs<DataRef<AnyPointer>>()));
		}
		
		auto dataSize = metadataResponse.getMetadata().getDataSize();
		
		// Check if the data already exist in the blob db
		{
			KJ_IF_MAYBE(pBlob, blobStore -> find(metadataResponse.getMetadata().getDataHash())) {
				refData.getDownloadStatus().setFinished();
				pBlob -> incRef();
				dst.save();
				return READY_NOW;
			}
		}
		
		refData.getDownloadStatus().setDownloading();
			
		auto downloadRequest = src.transmitRequest();
		downloadRequest.setStart(0);
		downloadRequest.setEnd(dataSize);
		downloadRequest.setReceiver(kj::heap<TransmissionReceiver>(dst));
		
		dst.save();
		
		return downloadRequest.send().ignoreResult()
		.then([this, &dst]() mutable {
			return withODBBackoff([this, &dst]() {
				auto t = conn -> beginTransaction();
				dst.load();
				
				auto refData = dst.info.getDataRef();
				refData.getDownloadStatus().setFinished();
				dst.save();
			});
		});
	});
	});
	});
}

void ObjectDB::deleteIfOrphan(int64_t id) {
	KJ_REQUIRE(!readOnly);
	
	auto rcAndHash = getRefcountAndHash.query(id);
	KJ_REQUIRE(rcAndHash.step(), "Internal error, refcount not found");
	if(rcAndHash[0].asInt64() > 0)
		return;
	
	auto hash = rcAndHash[1].asBlob();
	
	// Decrease refcount on blob
	KJ_IF_MAYBE(pBlob, blobStore -> find(hash)) {
		pBlob -> decRef();
	}
	
	// Scan outgoing refs
	std::set<int64_t> idsToCheck;
	{
		auto outgoingRefs = listOutgoingRefs.query(id);
		while(outgoingRefs.step()) {
			auto target = outgoingRefs[0].asInt64();
			decRefcount(target);
			idsToCheck.insert(id);
		}
	}
	
	deleteObject(id);
	
	// Check if we have capabilities without refs
	for(int64_t id : idsToCheck) {
		deleteIfOrphan(id);
	}
}

void DBObject::save() {
	KJ_REQUIRE(!parent -> readOnly);
	// All work here has to be done inside a DB transaction
	auto t = parent -> conn -> beginTransaction();
	
	// Now we need to figure out how to serialize references. This means we need to make a new message builder
	MallocMessageBuilder outputBuilder(info.totalSize().wordCount + 1);
	
	// Decrease refcount of all previous outgoing references
	std::set<int64_t> idsToCheck;
	{
		auto outgoingRefs = parent -> listOutgoingRefs.query(id);
		while(outgoingRefs.step()) {
			auto target = outgoingRefs[0].asInt64();
			parent -> decRefcount(target);
			idsToCheck.insert(id);
		}
	}
	
	// Clear existing references
	parent -> clearOutgoingRefs(id);
	
	// Compact info into a correctly sized message and capture the used capabilities
	BuilderCapabilityTable capTable;
	AnyPointer::Builder outputRoot = capTable.imbue(outputBuilder.initRoot<AnyPointer>());
	outputRoot.setAs<ObjectInfo>(info);
	
	// Serialize the message into the database
	kj::Array<byte> flatInfo = wordsToBytes(messageToFlatArray(outputBuilder));
	parent -> setInfo(id, flatInfo);
	
	// Iterate through the captured capabilities and store the links in the appropriate DB table
	kj::ArrayPtr<kj::Maybe<kj::Own<ClientHook>>> capTableData = capTable.getTable();
	for(auto i : kj::indices(capTableData)) {
		KJ_IF_MAYBE(pHook, capTableData[i]) {
			Capability::Client client((*pHook) -> addRef());
			auto unwrapped = parent -> unwrap(client);
			
			KJ_REQUIRE(!unwrapped.is<Capability::Client>(), "Only immediate DB references and null capabilities may be used inside DBObject.");
			if(unwrapped.is<Own<DBObject>>()) {
				auto& target = unwrapped.get<Own<DBObject>>();
				
				parent -> insertRef(id, i, target -> id);
				parent -> incRefcount(target -> id);
				
				continue;
			}
		}
		
		parent -> insertRef(id, i, nullptr);
	}
	
	// Check if we have capabilities without incoming refs
	for(int64_t id : idsToCheck) {
		parent -> deleteIfOrphan(id);
	}		
}

void DBObject::load() {
	// All work here has to be done inside a DB transaction
	auto t = parent -> conn -> beginTransaction();
	
	auto q = parent -> getInfo.query();
	KJ_REQUIRE(q.step(), "Object not present in database");
	
	auto heapBuffer = bytesToWords(kj::heapArray<const byte>(q[0].asBlob()));
	FlatArrayMessageReader reader(heapBuffer);
	
	kj::Vector<Maybe<Own<ClientHook>>> rawCapTable;
	
	auto refs = parent -> listOutgoingRefs.query(id);
	while(refs.step()) {
		auto col = refs[0];
		
		if(col.type() == sqlite::Type::NULLTYPE) {
			rawCapTable.add(nullptr);
			continue;
		}
		
		auto dbObject = kj::heap<DBObject>(*parent, col, CreationToken());
		rawCapTable.add(ClientHook::from(parent -> wrap(mv(dbObject))));
	}
	
	ReaderCapabilityTable capTable(rawCapTable.releaseAsArray());
	auto root = capTable.imbue(reader.getRoot<ObjectInfo>());
	
	infoHolder = kj::heap<MallocMessageBuilder>(root.totalSize().wordCount + 1);
	infoHolder -> setRoot(root);
	
	info = infoHolder -> getRoot<ObjectInfo>();
}

}}