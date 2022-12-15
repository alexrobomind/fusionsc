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

Object ObjectDB::store(AnyPointer ptr) {
	static_assert(false, "Store non-capability pointers in DataRefs");
	
	DBObject dbObject = storeInternal(object);
	return wrap(mv(dbObject));
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
	
	auto asCap = unwrapped.get<Capability::Client>();
	Own<DBObject> dbObject = downloadInternal(asCap.as<DataRef<AnyPointer>>());
	return wrap(mv(dbObject));
}

Own<DBObject> ObjectDB::downloadInternal(Capability::Client object) {	
	// Initialize the object to an unresolved state
	Own<DBObject> dbObject;
	{
		auto transaction = conn -> beginTransaction();
		dbObject = createObject();
		dbObject -> data.setUnresolved();
		dbObject -> save();
	}
	
	exports.put(ClientHook::from(cp(object)).get(), dbObject -> id);
	
	Promise<void> exportTask = downloadDatarefIntoDBObject(object.as<DataRef>(), *dbObject)
	.catch_([this, object, dbObject](kj::Exception& e) {
		if(e.getType() == kj::Exception::UNIMPLEMENTED)
			dbObject -> data.setUnknownObject();
		else
			dbObject -> data.setException(toProto(e));
		
		try {
			dbObject -> save();
		} catch(kj::Exception& e) {
			// Oh well, we tried, object will now likely have to stay in limbo
			// Fortunately, this case is ridiculously unlikely
		}
	})
	.then([this, object, id = dbObject -> id]() {
		// When the export task concludes, remove the
		// exported hook
		exports.remove(pHook);
		imports.remove(dbObject -> id);
	}).eagerlyEvaluate(nullptr);
	
	whenResolved.put(dbObject -> id, exportTask.fork());
	return dbObject;
}

Promise<void> ObjectDB::downloadDatarefIntoDBObject(DataRef<AnyPointer>::Client src, DBObject& dst) {
	using RemoteRef = DataRef<AnyPointer>;
	using MetadataResponse = Response<RemoteRef::MetadataResults>;
	using CapTableResponse = Response<RemoteRef::CapTableResults>;
	
	auto metadataPromise = src.metadataRequest().send().eagerlyEvaluate(nullptr);
	auto capTablePromise = src.capTableRequest().send().eagerlyEvaluate(nullptr);
	
	// I really need coroutines
	return metadataPromise
	.then([src, &dst, capTablePromise = mv(capTablePromise)](MetadataResponse metadataResponse) {
		return capTablePromise.then([src, &dst, metadataResponse = mv(metadataResponse)](CapTableResponse capTableResponse) -> Promise<void> {
			auto refData = dst.initResolved().initDataRef();
			
			refData.setMetadata(metadataResponse.getMetadata());
			
			// Copy cap table over, wrap child objects in their own download processes
			auto capTableIn = capTableResponse.getCapTable();
			auto capTableOut = refData.initCapTable(capTableIn.size());
			for(auto i : kj::indices(capTableIn)) {
				capTableOut.set(i, download(capTableIn[i]));
			}
			
			auto dataSize = metadataResponse.getMetadata().getDataSize();
			
			// Check if the data already exist in the blob db
			{
				auto t = db.connection -> beginTransaction();
				
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
			downloadRequest.setReceiver(kj::heap<ObjectRefReceiver>(dst));
			
			dst.save();
			
			return downloadRequest.send().ignoreResult()
			.then([this, refData, DBObject& dst]() {
				refData.getDownloadStatus().setFinished();
				dst.save();
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
	
	// Clear existing references
	parent -> clearOutgoingRefs(id);
	
	getInfo.reset();
	
	// Now we need to figure out how to serialize references. This means we need to make a new message builder
	MallocMessageBuilder outputBuilder(info.totalSize().sizeInWords() + 1);
	
	// Decrease refcount of all outgoing references
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

namespace {
	struct TransmissionProcess {
		constexpr static inline size_t CHUNK_SIZE = 1024 * 1024;
		
		BlobReader reader;
		
		DataRef<capnp::AnyPointer>::Receiver::Client receiver;
		size_t start;
		size_t end;
		
		Array<byte> buffer;
		
		TransmissionProcess(DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end) :
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
}


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
	
	Promise<ObjectInfo::Resolved::Builder> whenReady() {		
		auto exception = [this](capnp::rpc::Exception::Reader e) -> kj::Exception {
			return fromProto(e);
		};
		
		object -> load();
		auto entry = object -> data;
		switch(entry.which()) {
			case DBEntry::UNRESOLVED:
				return object -> whenUpdated().then([this]() { return whenReady(); });
			case DBEntry::EXCEPTION:
				return exception(entry.getException);
			case DBEntry::RESOLVED: {
				return entry.getResolved();
			}
			default:
				KJ_FAIL_REQUIRE("Unknown object type");
		}
	}
	
	Promise<Folder::Builder> whenFolder() {
		return whenReady().then([this](ObjectInfo::Resolved::Builder resolved) {
			if(resolved.which() != ObjectInfo::Resolved::FOLDER)
				KJ_UNIMPLEMENTED("Not a folder");
			
			return resolved.getFolder();
		});
	}
	
	Promise<StoredData::Builder> whenData() {
		return whenReady().then([this](ObjectInfo::Resolved::Builder resolved) {
			switch(resolved.which()) {
				case DBEntry::Resolved::DOWNLOADING:
					return object.whenUpdated().then([this]() { return whenData(); });
				case DBEntry::Resolved::DOWNLOAD_FAILED:
					return exception(resolved.getDownloadFailed());
				case DBEntry::Resolved::DOWNLOAD_SUCCEEDED:
					return READY_NOW;
				default:
					KJ_UNIMPLEMENTED("Not a DataRef");
			}
			
		});
	}
	
	Promise<void> getType(GetTypeContext ctx) override {
		return whenReady()
		.then([this](ObjectInfo::Resolved::Reader resolved) {
			Type type;
			switch(resolved.which()) {
				case ObjectInfo::Resolved::FOLDER:
					type = Type::FOLDER;
					break;
				case DBEntry::Resolved::DOWNLOADING:
				case DBEntry::Resolved::DOWNLOAD_FAILED:
				case DBEntry::Resolved::DOWNLOAD_SUCCEEDED:
					type = Type::DATA;
					break;
				default:
					KJ_FAIL_REQUIRE("Unknown self type");
			}
			
			ctx.getResults().setType(type);
		});
	}
	
	Promise<void> metadata(MetadataContext ctx) override {
		return whenData()
		.then([ctx](StoredData::Reader obj) {
			ctx.initResults().setMetadata(obj.getMetadata());
		});
	}
	
	Promise<void> rawBytes(RawBytesContext ctx) override {
		return whenData()
		.then([ctx](StoredData::Reader obj) {
			KJ_REQUIRE(end < obj.getMetadata().getDataSize());
			
			auto hash = obj.getMetadata().getHash();
			KJ_IF_MAYBE(pBlob, object -> parent -> blobDB.find(hash) {
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
				KJ_FAIL_REQUIRE("No associated blob in blob DB");
			}
		}
	}
	
	Promise<void> transmit(TransmitContext) override {
		static_assert(false, "Unimplemented");
	}
	
	Promise<void> capTable(CapTableContext) override {
		return whenData()
		.then([ctx](StoredData::Reader obj) {
			auto tableIn = obj.getCapTable();
			auto tableOut = ctx.initTable(tableIn.size());
			
			for(auto i : kj::indices(tableIn)) {
				if(tableIn[i] < 0) {
					tableOut[i] = nullptr;
					continue;
				}
				
				KJ_IF_MAYBE(pObj, object -> parent -> getObjectInternal(tableIn[i])) {
					tableOut = pObj -> asCapability();
				} else {
					tableOut = kj::Exception("Object not found in database");
				}
			}
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

}}