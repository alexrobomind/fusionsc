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

// ========================= ObjectDB ==============================

Object ObjectDB::store(AnyPointer ptr) {
	static_assert(false, "Store non-capability pointers in DataRefs");
	
	DBObject dbObject = storeInternal(object);
	return wrap(mv(dbObject));
}

Maybe<Own<DBObject>> ObjectDB::unwrap(Capability::Client object) {
	// Resolve to innermost hook
	// If any of the hooks encountered is an object hook, use that
	Own<capnp::ClientHook> hook = capnp::ClientHook::from(kj::cp(object));
	
	ClientHook* inner = hook.get();
	while(true) {
		if(inner -> getBrand() == ObjectHook::BRAND) {
			auto asObjectHook = static_cast<ObjectHook*>(inner);
			return asObjectHook -> object.addRef();
		}
	
		KJ_IF_MAYBE(pId, exports.find(inner)) {
			static_assert(false, "Do we want objects to be allowed to be re-exported? Think about mutability");
			return DBObject(*this, *pId);
		}
		
		KJ_IF_MAYBE(pHook, hook -> getResolved()) {
			hook = mv(*pHook);
		} else {
			break;
		}
	}
	
	return nullptr;
}

static_assert(false, "Maybe I should handle folders separately?. Mixing mutable and immutable objects seems like a recipe for disaster");

enum class ObjectType {
	UNKNOWN, DATA, FOLDER
};

Promise<ObjectType> determineType(Capability::Client cap) {
	// Check whether ObjectDB interface is supported
	auto asObject = cap.as<Object>();
	return asObject.getInfoRequest().send()
	.then([](Object::GetInfoResults results) { return results.getType(); })
	
	.catch_([cap](kj::Exception& e) {
		// If the object db interface is not supported, try to see whether the
		// DataRef interface is
		if(e.getType() == kj::Exception::UNIMPLEMENTED) {
			auto asRef = cap.as<DataRef>();
			
			return asRef.getMetadata().ignoreResult()
			.then([]() {
				return Object::Type::DATA;
			});
		}
		throw e;
	}
	
	// Check first if we are a DataRef
	.then([](Object::Type t) {
		switch(t) {
			case Object::Type::DATA: return ObjectType::DATA;
			case Object::Type::FOLDER: return ObjectType::FOLDER;
			default: return ObjectType::UNKNOWN;
		}
	})
	.catch_([cap](kj::Exception& e) {
		// If any UNIMPLEMENTED exception made it through, we don't know
		// what this is.
		if(e.getType() == kj::Exception::UNIMPLEMENTED) {
			return ObjectType::UNKNOWN;
		}
		throw e;
	}
}

Own<DBObject> ObjectDB::storeInternal(Capability::Client object) {
	// Check if we are trying to store one of our own objects
	// or a running export
	KJ_IF_MAYBE(pDBObj, unwrap(object)) {
		return mv(*pDBObj);
	}
	
	// Initialize the object to an unresolved state
	Own<DBObject> dbObject;
	{
		auto transaction = conn -> beginTransaction();
		dbObject = createObject();
		dbObject -> data.setUnresolved();
		dbObject -> save();
	}
	
	// From now on, this will keep the database object alive
	Capability::Client importClient = kj::heap<ObjectDBClient>(dbObject.x());
	
	//TODO: It would probably be better to attach the export to the inner most
	// client id.
	exports.put(ClientHook::from(cp(object)).get(), dbObject -> id);
	
	Promise<void> exportTask = determineType(object)
	.then([this, object, dbObject](ObjectType type) {
		switch(type) {
			case ObjectType::UNKNOWN {
				dbObject -> data.setUnknownObject();
				dbObject -> save();
				return;
			}
			
			case ObjectType::DATA {
				dbObject -> data.initDataRef().setDownloading();
				dbObject -> save();
				
				// Initiate download
				return downloadObject(dbObject, object.as<DataRef>());
				.catch_([this, object, dbObject](kj::Exception& e) {
					// Download failed
					static_assert("Should this not just set the global exception?");
					dbObject -> data.setDownloadFailed(toProto(e));
					dbObject -> save();
				});
			}
			
			case ObjectType::FOLDER {
				return downloadFolder(dbObject, object.as<Folder>());
			}
		}
	})
	.catch_([this, object, dbObject](kj::Exception& e) {
		// Object resolution or download failed, 
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