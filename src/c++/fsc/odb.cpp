#include "odb.h"

using kj::str;

using namespace capnp;

namespace fsc { namespace odb {
	
BlobStore::BlobStore(sqlite::Connection& connRef, kj::StringPtr tablePrefix) :
	tablePrefix(kj::heapString(tablePrefix)),
	conn(connRef.addRef())
{
	connRef.exec(str(
		"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_blobs ("
		"  id INTEGER PRIMARY KEY,"
		"  hash BLOB UNIQUE," // SQLite UNIQUE allows multiple NULL values
		"  externalRefcount INTEGER"
		// "  internalRefcount INTEGER"
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
	//incRefInternal = conn->prepare(str("UPDATE ", tablePrefix, "_blobs SET internalRefcount = internalRefcount + 1 WHERE id = ?"));
	//decRefInternal = conn->prepare(str("UPDATE ", tablePrefix, "_blobs SET internalRefcount = internalRefcount - 1 WHERE id = ?"));
	
	deleteIfOrphan = conn->prepare(str("DELETE FROM ", tablePrefix, "_blobs WHERE id = ? AND externalRefcount = 0")); // AND internalRefcount = 0
	
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
	KJ_REQUIRE(parent.conn -> inTransaction(), "Must be inside a transaction");
}

Blob::~Blob() {
	if(parent.get() == nullptr)
		return;
	
	ud.catchExceptionsIfUnwinding([this]() {
		// parent -> decRefInternal(id);
		// parent -> deleteIfOrphan(id);
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
	KJ_REQUIRE(parent -> conn.inTransaction(), "Must be inside transaction");
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

// ========================= ObjectDB ==============================

namespace {
	template<typename T>
	auto withODBBackoff(T func) {
		return withBackoff(10 * kj::MILLISECONDS, 5 * kj::MINUTES, 2, mv(func));
	}
	
	KJ_NORETURN void objectDeleted() {
		kj::throwFatalException(KJ_EXCEPTION(DISCONNECTED, "Object was deleted from database"));
	}
	
	struct TransmissionProcess {
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
			return request.send().ignoreResult().then([this, start]() { return transmit(start); });
		}
		
		Promise<void> transmit(size_t chunkStart) {			
			// Check if we are done transmitting
			if(chunkStart >= end)
				return receiver.doneRequest().send().ignoreResult();
			
			auto slice = chunkStart + CHUNK_SIZE <= end ? : buffer.asPtr() : buffer.slice(0, end - chunkStart);
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
	
	struct TransmissionReceiver : public DataRef<AnyPointer>::Receiver::Server {
		Own<BlobBuilder> builder;
		Own<DBObject> parent;
		
		TransmissionReceiver(DBObject& obj) :
			parent(obj.addRef())
		{}
		
		void begin(BeginContext ctx) {
			return withODBBackoff([this]() {
				// Always grab the write lock before doing anything to fail fast
				auto t = parent -> parent -> conn -> beginRootTransaction(true);
			
				builder = kj::heap<BlobBuilder>(parent -> parent -> blobStore);
			});
		}
		
		void receive(ReceiveContext ctx) {
			return withODBBackoff([this, ctx]() {
				// Always grab the write lock before doing anything to fail fast
				auto t = parent -> parent -> conn -> beginRootTransaction(true);
				builder -> write(ctx.getParams().getData());
			});
		}
		
		void done(DoneContext ctx) {
			return withODBBackoff([this]() {
				// Always grab the write lock before doing anything to fail fast
				auto t = parent -> parent -> conn -> beginRootTransaction(true);
				
				Blob blob = builder -> finish();
				
				obj.load();
				obj.getDataref().getMetadata().setDataHash(blob.hash());
				obj.save();
				blob.incRefExternal();
			});
		}
	};
	
	//! Marker hook to indicate that a capability is derived from a database object or being exported as such
	struct ObjectHook : public ClientHook, public kj::Refcounted {
		Own<ClientHook> inner;
		static const uint BRAND;
		DBObject& object;
		
		ObjectHook(Own<DBObject> objectIn) :
			inner(ClientHook::from(Capability::Client(kj::heap<ObjectImpl>(*objectIn)))),
			object(*objectIn)
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
			return inner -> newCall(interfaceId, methodId, mv(context), mv(hints));
		}

		kj::Maybe<ClientHook&> getResolved() override {
			return *inner;
		}

		kj::Maybe<kj::Promise<kj::Own<ClientHook>>> whenMoreResolved() override {
			return nullptr;
		}

		virtual kj::Own<ClientHook> addRef() override { return kj::addRef(*this); }

		const void* getBrand() override { return OBJECT_CAPABILITY_BRAND; }

		kj::Maybe<int> getFd() override { return nullptr; }
	}

	struct ObjectImpl : public Object::Server {
		Own<DBObject> object;
		
		using StoredData = ObjectInfo::Resolved::DownloadSucceeded;
		
		//! Waits until this object has settled into a usable state
		Promise<void> whenReady() {
			return withODBackoff([this]() {
				object -> load(); 
				auto info = object -> info;
				switch(info.which()) {
					case ObjectInfo::UNRESOLVED:
						return object -> whenUpdated().then([this]() { return whenReady(); });
					case ObjectInfo::EXCEPTION:
						return fromProto(info.getException());
						
					default:
						return READY_NOW;
				});
				
				return info.getFolder();
			})
		}
		
		Promise<void> whenDownloadReady() {
			return whenReady()
			.then([this}() {
				return withODBBackoff([this]() {		
					switch(checkRef().getDownloadStatus()) {
						case ObjectInfo::DataRef::DOWNLOADING:
							return object -> whenUpdated().then([this]() { return whenDownloadReady(); });
						case ObjectInfo::DataRef::FINISHED:
							return READY_NOW;
					}
				});
			});
		}
		
		ObjectInfo::Folder::Builder checkFolder() {
			if(object -> info.which() != ObjectInfo::FOLDER)
				KJ_UNIMPLEMENTED("This database object is not a folder");
		}
		
		ObjectInfo::DataRef::Builder checkRef() {
			if(object -> info.which() != ObjectInfo::DATA_REF)
				KJ_UNIMPLEMENTED("This database object is not a DataRef");
		}
		
		Promise<void> getType(GetTypeContext ctx) override {
			return whenReady()
			.then([this](ObjectInfo::Resolved::Reader resolved) {
				Type type;
				auto info = object -> info;
				
				switch(info.which()) {
					case ObjectInfo::FOLDER:
						type = Type::FOLDER;
						break;
					case ObjectInfo::DATA_REF:
						type = Type::DATA;
						break;
					default:
						KJ_FAIL_REQUIRE("Internal error");
				});
				
				ctx.getResults().setType(type);
			});
		}
		
		Promise<void> metadata(MetadataContext ctx) override {
			// The metadata table is only ready once the hash is verified
			return whenDownloadReady()
			.then([this, ctx]() {
				ctx.initResults().setMetadata(checkRef().getMetadata());
			});
		}
		
		Promise<void> capTable(CapTableContext) override {
			return whenReady()
			.then([this, ctx]() {
				return withODBBackoff([this, ctx]() {
					auto t = object -> parent -> conn -> beginTransaction();
					ctx.getResults().setCapTable(checkRef().getCapTable());
				});
			});
		}
		
		Promise<void> rawBytes(RawBytesContext ctx) override {
			return whenDownloadReady()
			.then([this, ctx]() {
				return withODBBackoff([this, ctx]() {
					auto t = object -> parent -> readConn -> beginTransaction();
					
					auto refInfo = checkRef();
					KJ_REQUIRE(end < refInfo.getMetadata().getDataSize());
					
					auto hash = refInfo.getMetadata().getDataHash();
					KJ_IF_MAYBE(pBlob, object -> parent -> blobDB.find(hash)) {
						auto buffer = kj::heapArray<byte>(8 * 1024 * 1024);
						reader = obj.read();
						
						const uint64_t start = ctx.getParams().getStart();
						const uint64_t end = ctx.getParams().getEnd();
						KJ_REQUIRE(end >= start);
						
						// Wind forward the stream until we hit the target point
						uint64_t windRemaining = start;
						while(true) {
							if(remaing >= buffer.size()) {
								reader.read(buffer);
								windRemaining -= buffer.size();
							} else {
								reader.read(buffer.slice(0, windRemaining));
								break;
							}
						}
						
						auto data = ctx.getResults().initData(end - start);
						obj.read(data);
					} else {
						objectDeleted();
					}
				});
			}
		}
		
		Promise<void> transmit(TransmitContext) override {
			return whenDownloadReady()
			.then([this, ctx]() {
				return withODBBackoff([this, ctx]() {
					auto params = ctx.getParams();
					
					// Since we preparing a long-running op, fork the blob store so that our main connection doesn't get blocked
					auto forkedStore = kj::refcounted<BlobStore>(forkConnection(true), tablePrefix);
					auto forkedTransaction = forkedStore -> conn -> beginTransaction();
					
					// Note: Read operations can still fail					
					KJ_IF_MAYBE(pBlob, object -> parent -> blobStore.find(checkRef().getMetadata().getDataHash())) {
						auto reader = pBlob -> open();
						auto transProc = heapHeld<TransmissionProcess>(params.getReceiver(), params.getBegin(), params.getEnd());
						return transProc.run().attach(mv(forkedStore), mv(forkedTransaction));
					} else {
						objectDeleted();
					}
				});
			});
		}
		
		Promise<void> ls(LsContext ctx) override {
			return whenFolder()
			.then([ctx](Folder::Reader f) {
				auto in = f.getEntries();
				auto out = ctx.getResults().initEntries(in.size());
				for(auto i : kj::indices(in)) {
					out[i] = in[i].getName();
				}
			});
		}
		
		Promise<void> getAll(GetAllContext ctx) override {
			return whenFolder()
			.then([ctx](Folder::Reader f) {
				ctx.getResults().setEntries(f.getEntries);
			});
		}
		
		Promise<void> getObject(GetObjectContext ctx) override {
			return whenFolder()
			.then([ctx](Folder::Reader f) {
				auto name = ctx.getParams().getName();
				for(auto e : f.getEntries()) {
					if(e.getName() == name) {
						ctx.getResults().setObject(e.getRef());
						break;
					}
				}
			});
		}
		
		Promise<void> setObject(GetObjectContext ctx) override {
			return whenFolder()
			.then([ctx](Folder::Builder f) {
				auto name = ctx.getParams().getName();
				auto ref = ctx.getParams().getObject();
				
				for(auto e : f.getEntries()) {
					if(e.getName() == name) {
						e.setRef(ref);
						return;
					}
				}
				
				auto orphanage = Orphanage::getForMessageContaining(f);
				auto newList = orphanage.newOrphan<Folder::Entry>(1);
				newList[0].setName(name);
				newList[1].setRef(ref);
				
				f.adoptEntries(orphanage.newOrphanConcat<Folder::Entry>({f.getEntries(), newList}));
				object -> save();
			})
		}
	};
}

Object ObjectDB::store(AnyPointer ptr) {
	static_assert(false, "Store non-capability pointers in DataRefs");
	
	DBObject dbObject = storeInternal(object);
	return wrap(mv(dbObject));
}

Capability::Client wrap(Own<DBObject> object) {
	Capability::Client innerClient(kj::heap<ObjectImpl>(mv(object)));
	
	Own<ClientHook> wrapper = kj::heap<ObjectHook>(ClientHook::from(mv(innerClient)));
	return Capability::Client(mv(wrapper));
}

OneOf<Capability::Client, Own<DBObject>, decltype(nullptr)> ObjectDB::unwrap(Capability::Client object) {
	// Resolve to innermost hook
	// If any of the hooks encountered is an object hook, use that
	Own<capnp::ClientHook> hook = capnp::ClientHook::from(kj::cp(object));
	
	ClientHook* inner = hook.get();
	while(true) {
		if(inner -> getBrand() == ObjectHook::BRAND) {
			auto asObjectHook = static_cast<ObjectHook*>(inner);
			
			if(asObjectHook -> object -> parent.get() == this)
				return asObjectHook -> object.addRef();
		}
	
		KJ_IF_MAYBE(pId, exports.find(inner)) {
			return kj::refcounted<DBObject>(*this, *pId);
		}
		
		KJ_IF_MAYBE(pHook, inner -> getResolved()) {
			inner = pHook;
		} else {
			break;
		}
	}
	
	if(inner -> isNull())
		return nullptr;
	}
	
	return mv(object);
}

DataRef<AnyPointer>::Client ObjectDB::download(DataRef<AnyPointer>::Client object) {
	auto unwrapped = unwrap(object);
	
	if(unwrapped.is<decltype(nullptr)>()) {
		return nullptr;
	}
	
	if(unwrapped.is<Own<DBObject>>()) {
		return wrap(mv(unwrapped.get<Own<DBObject>>()));
	}
	
	return withODBBackoff([unwrapped = mv(unwrapped), this]() {
		auto asCap = unwrapped.get<Capability::Client>();
		Own<DBObject> dbObject = startDownloadTask(asCap.as<DataRef<AnyPointer>>());
		return wrap(mv(dbObject));
	}).attach(thisCap());
}

Own<DBObject> ObjectDB::startDownloadTask(DataRef<AnyPointer>::Client object) {	
	// Initialize the object to an unresolved state
	Own<DBObject> dbObject;
	{
		auto transaction = conn -> beginTransaction();
		dbObject = createObject();
		dbObject -> data.setUnresolved();
		dbObject -> save();
	}
	
	// Remember that we are downloading this capability into the given address
	// This will prevent starting double-downloads until the hash is actually
	// present in our blob db.
	activeDownloads.put(ClientHook::from(cp(object)).get(), dbObject -> id);
	
	// Start the download task
	ForkedPromise<void> exportTask = downloadTask(object.as<DataRef>(), *dbObject)
	
	// If the download fails, store the failure
	.catch_([this, object, dbObject](kj::Exception& e) {
		return withODBBackoff([this, e]() {
			if(e.getType() == kj::Exception::UNIMPLEMENTED)
				dbObject -> data.setUnknownObject();
			else
				dbObject -> data.setException(toProto(e));
			
			dbObject -> save();
		});
	})
	// If the failure storage also fails, discard the error, database is probably
	// having larger issues right now anyway
	.catch_([](kj::Exception& e) {})
	
	// Remove exported hook after conclusion
	.then([this, object, id = dbObject -> id]() {
		activeDownloads.remove(pHook);
		whenResolved.remove(dbObject -> id);
	})
	.fork();
	
	whenResolved.put(dbObject -> id, exportTask.addBranch());
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
	.then([src, &dst, capTablePromise = mv(capTablePromise)](MetadataResponse metadataResponse) {
	return capTablePromise.then([src, &dst, metadataResponse = mv(metadataResponse)](CapTableResponse capTableResponse) {
	return withBackoff([src, &dst, metadataResponse = mv(metadataResponse), capTableResponse = mv(capTableResponse)]() -> Promise<void> {
		auto t = db.connection -> beginTransaction();
		
		auto refData = dst.initResolved().initDataRef();
		
		refData.setMetadata(metadataResponse.getMetadata());
		refData.getMetadata().setDataHash(nullptr);
		
		// Copy cap table over, wrap child objects in their own download processes
		auto capTableIn = capTableResponse.getCapTable();
		auto capTableOut = refData.initCapTable(capTableIn.size());
		for(auto i : kj::indices(capTableIn)) {
			capTableOut.set(i, download(capTableIn[i]));
		}
		
		auto dataSize = metadataResponse.getMetadata().getDataSize();
		
		// Check if the data already exist in the blob db
		{
			KJ_IF_MAYBE(pBlob, db.blobStore.find(metadataResponse.getMetadata().getDatahash())) {
				refData.getDownloadStatus().setFinished();
				pBlob -> incRefExternal();
				dst.save();
				return READY_NOW;
			}
		}
		
		refData.getDownloadStatus().setDownloading();
			
		auto downloadRequest = refData.transmitRequest();
		downloadRequest.setStart(0);
		downloadRequest.setEnd(dataSize);
		downloadRequest.setReceiver(kj::heap<TransmissionReceiver>(dst));
		
		dst.save();
		
		return downloadRequest.send().ignoreResult()
		.then([this, refData, &dst]() {
			return withBackoff([this, refData, &dst]() {
				refData.getDownloadStatus().setFinished();
				dst.save();
			});
		});
	});
	});
	});
}

void ObjectDB::deleteIfOrphan(int64_t id) {
	KJ_REQUIRE(parent -> getRefcountAndHash.query(id), "Internal error, refcount not found");
	if(parent -> getRefcountAndHash[0] > 0)
		return;
	
	auto hash = parent -> getRefcountAndHash[1];
	
	// Decrease refcount on blob
	KJ_IF_MAYBE(pBlob, blobStore -> find(hash)) {
		pBlob -> decRefExternal();
	}
	
	// Scan outgoing refs
	std::set<int64_t> idsToCheck;
	{
		auto& getOutgoingRefs = parent -> getOutgoingRefs;
		getOutgoingRefs.bind(id);
		getOutgoingRefs.reset();
		while(getOutgoingRefs.step()) {
			auto target = getOutgoingRefs[0].asInt64();
			parent -> decRefcount(target);
			idsToCheck.insert(id);
		}
	}
	
	parent -> deleteObject(id);
	
	// Check if we have capabilities without refs
	for(int64_t id : idsToCheck) {
		deleteIfOrphan(id);
	}
}

void DBObject::save() {
	// All work here has to be done inside a DB transaction
	auto t = parent -> conn.beginTransaction();
	
	// Now we need to figure out how to serialize references. This means we need to make a new message builder
	MallocMessageBuilder outputBuilder(info.totalSize().sizeInWords() + 1);
	
	// Decrease refcount of all previous outgoing references
	std::set<int64_t> idsToCheck;
	{
		auto& getOutgoingRefs = parent -> getOutgoingRefs;
		getOutgoingRefs.bind(id);
		getOutgoingRefs.reset();
		while(getOutgoingRefs.step()) {
			auto target = getOutgoingRefs[0].asInt64();
			parent -> decRefcount(target);
			idsToCheck.insert(id);
		}
	}
	
	// Clear existing references
	parent -> clearOutgoingRefs(id);
	
	// Compact info into a correctly sized message and capture the used capabilities
	BuilderCapabilityTable capTable;
	AnyPointer::Builder outputRoot = capTable.imbue(outputBuilder.initRoot<AnyPointer>());
	outputRoot.setAs(info);
	
	// Serialize the message into the database
	kj::Array<byte> flatInfo = messageToflatArray(outputBuilder).asBytes();
	parent -> setInfo(id, flatInfo);
	
	// Iterate through the captured capabilities and store the links in the appropriate DB table
	kj::ArrayPtr<kj::Maybe<kj::Own<ClientHook>>> capTableData = capTable.getTable();
	for(auto i : kj::indices(capTableData)) {
		KJ_IF_MAYBE(pHook, capTableData[i]) {
			Capability::Client client(pHook -> addRef());
			auto unwrapped = unwrap(client);
			
			KJ_REQUIRE(!unwrapped.is<Capability::Client>(), "Only immediate DB references and null capabilities may be used inside DBObject.");
			if(unwrapped.is<Own<DBObject>>()) {
				auto& target = unwrapped.as<Own<DBObject>>();
				parent -> insertRef(id, i, target.id);
				parent -> incRefcount(target.id);
				continue;
			}
		}
		
		parent -> insertRef(id, i, nullptr);
	}
	
	// Check if we have capabilities without refs
	for(int64_t id : idsToCheck) {
		parent -> deleteIfOrphan(id);
	}		
}

void DBObject::load() {
	// All work here has to be done inside a DB transaction
	auto t = parent -> conn.beginTransaction();
	
	KJ_REQUIRE(getInfo.query(id), "Object not present in database");
	
	auto heapBuffer = bytesToWords(kj::heapArray<const byte>(getInfo[0].asBlob()));
	FlatArrayMessageReader reader(heapBuffer);
	
	kj::Vector<Maybe<Own<ClientHook>>> rawCapTable;
	
	parent -> getRefs.reset();
	parent -> getRefs.bind(id);
	while(parent -> getRefs.step()) {
		auto col = parent -> getRefs[0];
		
		if(col.type() == sqlite::Type::NULLTYPE) {
			rawCapTable.add(nullptr);
			continue;
		}
		
		int64_t id = parent -> getRefs[0];
		auto dbObject = kj::heap<DBObject>(*parent, id, CreationToken());
		rawCapTable.add(ClientHook::from(wrap(mv(dbObject))));
	}
	
	ReaderCapabilityTable capTable(rawCapTable.releaseAsArray());
	auto root = capTable.imbue(reader.getRoot<ObjectInfo>());
	
	infoHolder = kj::heap<MallocMessageBuilder>(root.totalSize().sizeInWords() + 1);
	infoHolder.setRoot(root);
	
	info = infoHolder.getRoot<ObjectInfo>();
}

}}