#include "odb.h"
#include "data.h"

#include <capnp/rpc.capnp.h>

#include <set>

using kj::str;

using namespace capnp;

namespace fsc { namespace odb {

/*
Folder::Client openObjectDB(kj::StringPtr folder) {
	auto objectDB = kj::refcounted<ObjectDB>(folder, "objectdb", false);
	
	auto clientHook = ClientHook::from(objectDB -> getRoot());
	clientHook = clientHook.attach(kj::defer([db = mv(objectDB)]() mutable {
		db -> cancelDownloads();
	}));
	
	return Folder::Client(mv(clientHook));
}*/

// ====================== class BlobStore =======================
	
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
			"  refcount INTEGER DEFAULT 0"
			")"
		));
		conn.exec(str(
			"CREATE TABLE IF NOT EXISTS ", tablePrefix, "_chunks ("
			"  id INTEGER,"
			"  chunkNo INTEGER,"
			"  data BLOB,"
			""
			"  PRIMARY KEY(id, chunkNo),"
			"  FOREIGN KEY(id) REFERENCES ", tablePrefix, "_blobs(id) ON UPDATE CASCADE ON DELETE CASCADE"
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
	getBlobHash = conn.prepare(str("SELECT hash FROM ", tablePrefix, "_blobs WHERE id = ?"));
	getRefcount = conn.prepare(str("SELECT refcount FROM ", tablePrefix, "_blobs WHERE id = ?"));
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

int64_t Blob::refcount() {
	auto q = parent -> getRefcount.query(id);
	KJ_REQUIRE(q.step());
	
	return q[0].asInt64();
}


kj::Array<const byte> Blob::hash() {
	auto q = parent -> getBlobHash.query(id);
	
	if(!q.step())
		return nullptr;
	
	return kj::heapArray<byte>(q[0].asBlob());
}

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
		parent -> deleteIfOrphan(id);
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
	readStatement(blob.parent -> conn -> prepare(str("SELECT data FROM ", blob.parent -> tablePrefix, "_chunks WHERE id = ? ORDER BY chunkNo"))),
	readQuery(readStatement.query(blob.id))
{
}

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
	
	KJ_NORETURN (void objectDeleted());
	void objectDeleted() {
		kj::throwFatalException(KJ_EXCEPTION(DISCONNECTED, "Object was deleted from database"));
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
}

struct TransmissionProcess {
	constexpr static inline size_t CHUNK_SIZE = 1024 * 1024;
	
	Own<BlobReader> reader;
	
	DataRef<capnp::AnyPointer>::Receiver::Client receiver;
	size_t start;
	size_t end;
	
	Array<byte> buffer;
	
	TransmissionProcess(Own<BlobReader>&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end) :
		reader(mv(reader)),
		receiver(mv(receiver)),
		buffer(kj::heapArray<byte>(CHUNK_SIZE)),
		start(start), end(end)
	{
		KJ_REQUIRE(end >= start);
	}
	
	Promise<void> run() {
		// Wind forward the stream until we hit the target point
		uint64_t windRemaining = start;
		while(true) {
			if(windRemaining >= buffer.size()) {
				reader -> read(buffer);
				windRemaining -= buffer.size();
			} else {
				reader -> read(buffer.slice(0, windRemaining));
				break;
			}
		}
		
		auto request = receiver.beginRequest();
		request.setNumBytes(end - start);
		return request.send().ignoreResult().then([this]() { return transmit(start); });
	}
	
	Promise<void> transmit(size_t chunkStart) {			
		// Check if we are done transmitting
		if(chunkStart >= end)
			return receiver.doneRequest().send().ignoreResult();
		
		auto slice = chunkStart + CHUNK_SIZE <= end ? buffer.asPtr() : buffer.slice(0, end - chunkStart);
		reader -> read(slice);
		KJ_REQUIRE(reader -> remainingOut() == 0, "Buffer should be filled completely");
		
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

/**
 * Transmission receiver that forwards the data into a blob store and in the end attaches
 * the resulting blob to a DBObject
 */
struct ObjectDB::TransmissionReceiver : public DataRef<AnyPointer>::Receiver::Server {
	Own<BlobBuilder> builder;
	Own<DBObject> parent;
	
	TransmissionReceiver(DBObject& obj) :
		parent(obj.addRef())
	{}
	
	Promise<void> begin(BeginContext ctx) override {
		return withODBBackoff([this]() {
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> getParent().conn -> beginRootTransaction(true);
			builder = kj::heap<BlobBuilder>(*(parent -> parent -> blobStore));
		});
	}
	
	Promise<void> receive(ReceiveContext ctx) override {
		return withODBBackoff([this, ctx]() mutable {
			KJ_REQUIRE(builder.get() != nullptr);
			
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> getParent().conn -> beginRootTransaction(true);
			
			builder -> write(ctx.getParams().getData());
		});
	}
	
	Promise<void> done(DoneContext ctx) override {
		return withODBBackoff([this]() {
			KJ_REQUIRE(builder.get() != nullptr);
			
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> getParent().conn -> beginRootTransaction(true);
			
			Blob blob = builder -> finish();
			blob.incRef();
			
			parent -> load();
			parent -> info.getDataRef().getMetadata().setDataHash(blob.hash());
			parent -> save();
		});
	}
};


struct DBCache::CachedRef : public DataRef<AnyPointer>::Server {
	Temporary<DataRefMetadata> _metadata;
	kj::Array<capnp::Capability::Client> refs;
	Blob blob;
	kj::UnwindDetector ud;
	Own<DBCache> parent;
	
	//! Create a cached reference by stealing metadata, refs AND blob (so no incRef inside, but a decRef on destruction)
	CachedRef(Temporary<DataRefMetadata>&& metadata, kj::Array<capnp::Capability::Client> refs, Blob blob, DBCache& parent) :
		_metadata(mv(metadata)),
		refs(mv(refs)),
		blob(blob),
		parent(parent.addRef())
	{
		// No incRef, since the constructor steals the blob's refcount
	}
	
	~CachedRef() {
		ud.catchExceptionsIfUnwinding([this]() mutable {
			auto t = blob.parent -> conn -> beginTransaction();
			blob.decRef();
		});
	}
	
	Promise<void> metaAndCapTable(MetaAndCapTableContext ctx) override {
		// The metadata table is only ready once the hash is verified
		ctx.initResults().setMetadata(_metadata);
		
		auto tableOut = ctx.getResults().initTable(refs.size());
		for(auto i : kj::indices(refs))
			tableOut.set(i, refs[i]);
		
		return READY_NOW;
	}
	
	Promise<void> rawBytes(RawBytesContext ctx) override {
		return withODBBackoff([this, ctx]() mutable {
			auto t = blob.parent -> conn -> beginTransaction();
				
			const uint64_t start = ctx.getParams().getStart();
			const uint64_t end = ctx.getParams().getEnd();
			KJ_REQUIRE(end >= start);
			KJ_REQUIRE(end < _metadata.getDataSize());
			
			auto buffer = kj::heapArray<byte>(8 * 1024 * 1024);
			auto reader = blob.open();
				
			// Wind forward the stream until we hit the target point
			uint64_t windRemaining = start;
			while(true) {
				if(windRemaining >= buffer.size()) {
					reader -> read(buffer);
					windRemaining -= buffer.size();
				} else {
					reader -> read(buffer.slice(0, windRemaining));
					break;
				}
			}
				
			auto data = ctx.getResults().initData(end - start);
			reader -> read(data);
		});
	}
	
	Promise<void> transmit(TransmitContext ctx) override {
		return withODBBackoff([this, ctx]() mutable {
			auto params = ctx.getParams();
			auto reader = blob.open();
			auto transProc = heapHeld<TransmissionProcess>(mv(reader), params.getReceiver(), params.getStart(), params.getEnd());
			
			auto transmission = kj::evalNow([=]() mutable { return transProc -> run(); });
			return transmission.attach(transProc.x());
		})
		.attach(thisCap());
	}
};

struct DBCache::DownloadProcess : public internal::DownloadTask<DataRef<capnp::AnyPointer>::Client> {
	Own<DBCache> parent;
	
	Own<BlobBuilder> builder;
	Maybe<Blob> blob;
	
	kj::UnwindDetector ud;
	
	DownloadProcess(DBCache& parent, DataRef<AnyPointer>::Client src) :
		DownloadTask(src, Context()),
		parent(parent.addRef())
	{}
	
	~DownloadProcess() {
		ud.catchExceptionsIfUnwinding([this]() {
			KJ_IF_MAYBE(pBlob, blob) {	
				auto t = parent -> store -> conn -> beginRootTransaction(true);
				pBlob -> decRef();
			}
		});
	}
	
	//! Check whether "src" can be directly unwrapped
	Promise<Maybe<ResultType>> unwrap() override {
		return parent -> SERVER_SET.getLocalServer(src)
		.then([this](Maybe<DataRef<AnyPointer>::Server> maybeResult) -> Maybe<ResultType> {
			KJ_IF_MAYBE(pResult, maybeResult) {
				auto backend = static_cast<CachedRef*>(pResult);
				
				// Ensure that the backend comes from the same store
				// TODO: I should just have a non-static server set here ...
				if(backend -> parent.get() != parent.get())
					return nullptr;
				
				return src;
			}
			
			return nullptr;
		});
	}
	
	//! Adjust refs e.g. by performing additional downloads. If the resulting client is broken with an exception of type "unimplemented", the original ref is used instead.
	capnp::Capability::Client adjustRef(capnp::Capability::Client ref) override {
		return mv(ref); //DBCache doesn't do recursive downloads
		/*if(!recursive)
			return mv(ref);
		
		return parent -> cache(ref.castAs<DataRef<AnyPointer>>(), true);*/
	}
	
	Promise<Maybe<ResultType>> useCached() override {
		return withODBBackoff([this]() mutable -> Maybe<ResultType> {		
			auto t = parent -> store -> conn -> beginRootTransaction(true);		
			
			// Check if blob is already present in store
			auto hash = metadata.getDataHash();
			KJ_IF_MAYBE(pBlob, parent -> store -> find(hash)) {
				pBlob -> incRef();
				
				auto server = kj::heap<CachedRef>(mv(metadata), mv(capTable), mv(*pBlob), *parent);
				auto client = parent -> SERVER_SET.add(mv(server));
				return client;
			}
			
			builder = kj::heap<BlobBuilder>(*parent -> store);
			return nullptr;
		});
	}
	
	Promise<void> beginDownload() override { return READY_NOW; }
	Promise<void> receiveData(kj::ArrayPtr<const byte> data) override {
		return withODBBackoff([this, data]() mutable {
			KJ_ASSERT(builder.get() != nullptr);
			
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> conn -> beginRootTransaction(true);
			builder -> write(data);
		});
	};
	Promise<void> finishDownload() override {
		return withODBBackoff([this]() mutable {
			KJ_ASSERT(builder.get() != nullptr);
			
			// Always grab the write lock before doing anything to fail fast
			auto t = parent -> conn -> beginRootTransaction(true);
			
			auto finishedBlob = builder -> finish();
			this -> blob = finishedBlob;
			finishedBlob.incRef();
			
			builder = nullptr;
		});
	}
	
	Promise<ResultType> buildResult() override {
		return withODBBackoff([this]() mutable {
			auto t = parent -> store -> conn -> beginRootTransaction(true);
			
			KJ_IF_MAYBE(pBlob, this -> blob) {
				auto server = kj::heap<CachedRef>(mv(metadata), mv(capTable), *pBlob, *parent);
				// The CachedRef steals the blob, so we need to set blob to
				// nullptr to avoid the destructor double-freeing
				this -> blob = nullptr;
				
				return parent -> SERVER_SET.add(mv(server));
			}
			
			KJ_FAIL_REQUIRE("Transmission process completed, but no blob stored");
		});
	}
};

Promise<DataRef<AnyPointer>::Client> DBCache::cache(DataRef<AnyPointer>::Client src) {
	auto downloadTask = kj::refcounted<DownloadProcess>(*this, mv(src));
	return downloadTask -> output();
}

capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>> DBCache::SERVER_SET;

/** Server class that gives clients access to DB object.
 *
 * \warning This class must be created on an already-loaded and resolved object.
 *          Using unresolved objects is not allowed (which is why this class should
 *          be instantiated through ObjectHook, which manages the resolution and
 *          redirection process.
 */
struct ObjectDB::ObjectImpl : public Object::Server {
	Own<DBObject> object;
	
	ObjectImpl(Own<DBObject>&& object) :
		object(mv(object))
	{}
	
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
	
	Promise<void> dataReady() {
		auto refInfo = checkRef();
		if(refInfo.getDownloadStatus().isFinished())
			return READY_NOW;
		
		return object -> whenUpdated()
		.then([this]() mutable {
			return withODBBackoff([this]() mutable {
				auto t = object -> parent -> conn -> beginTransaction();
				object -> load();
				
				return dataReady();
			});
		});
	}
	
	Promise<void> getInfo(GetInfoContext ctx) override {
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
		return READY_NOW;
	}
	
	Promise<void> metaAndCapTable(MetaAndCapTableContext ctx) override {
		// The metadata table is only ready once the hash is verified
		ctx.initResults().setMetadata(checkRef().getMetadata());
		
		return withODBBackoff([this, ctx]() mutable {
			auto t = object -> parent -> conn -> beginTransaction();
			
			auto tableIn = checkRef().getCapTable();
			auto tableOut = ctx.getResults().initTable(tableIn.size());
			for(auto i : kj::indices(tableIn))
				tableOut.set(i, tableIn[i]);
		});
	}
	
	Promise<void> rawBytes(RawBytesContext ctx) override {
		return dataReady()
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
							reader -> read(buffer);
							windRemaining -= buffer.size();
						} else {
							reader -> read(buffer.slice(0, windRemaining));
							break;
						}
					}
					
					auto data = ctx.getResults().initData(end - start);
					reader -> read(data);
				} else {
					objectDeleted();
				}
			});
		});
	}
	
	Promise<void> transmit(TransmitContext ctx) override {
		return dataReady()
		.then([this, ctx]() mutable {
			return withODBBackoff([this, ctx]() mutable {
				auto params = ctx.getParams();
				
				// Since we preparing a long-running op, fork the blob store so that our main connection doesn't get blocked
				// and the object doesn't get deleted during the transfer.
				// If the database isn't shared, we can use the main connection (though we have to hope the object doesn't get deleted
				auto forkedStore = kj::refcounted<BlobStore>(
					*(object -> parent -> forkConnection(true)),
					object -> parent -> tablePrefix,
					true
				);
				auto forkedTransaction = forkedStore -> conn -> beginTransaction();
				
				// Note: Read operations can still fail					
				KJ_IF_MAYBE(pBlob, forkedStore -> find(checkRef().getMetadata().getDataHash())) {
					auto reader = pBlob -> open();
					auto transProc = heapHeld<TransmissionProcess>(mv(reader), params.getReceiver(), params.getStart(), params.getEnd());
					
					auto transmission = kj::evalNow([=]() mutable { return transProc -> run(); });
					return transmission.attach(mv(forkedStore), mv(forkedTransaction), transProc.x());
				} else {
					objectDeleted();
				}
			});
		});
	}
	
	Promise<void> ls(LsContext ctx) override {
		object -> load();
		
		auto f = checkFolder();
		auto in = f.getEntries();
		auto out = ctx.getResults().initEntries(in.size());
		for(auto i : kj::indices(in)) {
			out[i] = in[i].getName();
		}
		
		return READY_NOW;
	}
	
	Promise<void> getAll(GetAllContext ctx) override {
		object -> load();
		
		auto f = checkFolder();
		ctx.getResults().setEntries(f.getEntries());
		
		return READY_NOW;
	}
	
	Promise<void> getEntry(GetEntryContext ctx) override {
		object -> load();
		
		auto f = checkFolder();
		auto name = ctx.getParams().getName();

		for(auto e : f.getEntries()) {
			if(e.getName().asString() == name) {
				ctx.setResults(e);
			}
		}
		
		return READY_NOW;
	}
	
	Promise<void> putEntry(PutEntryContext ctx) override {
		return withODBBackoff([this, ctx]() mutable -> Promise<void> {
			auto t = object -> parent -> conn -> beginTransaction();
			object -> load();
			
			auto f = checkFolder();
			
			auto params = ctx.getParams();
			auto name = params.getName();
			
			// Check if entry name contains a folder
			// If it does, make parent directory and call insert
			// on that.
			KJ_IF_MAYBE(pIdx, name.findLast('/')) {
				auto containingFolder = str(name.slice(0, *pIdx));
				
				auto mkdirRequest = thisCap().mkdirRequest();
				mkdirRequest.setName(containingFolder);
				
				auto newFolder = mkdirRequest.send().getFolder();
				auto tail = newFolder.putEntryRequest();
				tail.setName(str(name.slice(*pIdx + 1, name.size())));
				
				if(params.isFolder()) {
					tail.setFolder(params.getFolder());
				} else if(params.isRef()) {
					tail.setRef(params.getRef());
				}
				
				return ctx.tailCall(mv(tail));
			}
			
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
			
			auto retEntry = ctx.initResults();
			retEntry.setName(entry.getName());
			
			switch(params.which()) {
				case FolderEntry::REF:
					entry.setRef(object -> parent -> download(params.getRef()));
					
					retEntry.setRef(entry.getRef());
					break;
				
				case FolderEntry::FOLDER:
					auto unwrapped = object -> parent -> unwrap(params.getFolder());
					
					KJ_REQUIRE(unwrapped.is<Own<DBObject>>(), "Only resolved folders from this database may be passed to putEntry");
					entry.setFolder(params.getFolder());
					
					retEntry.setFolder(entry.getFolder());
					break;
			}
			
			object -> save();
			return READY_NOW;
		});
	}
	
	Promise<void> rm(RmContext ctx) override {
		return withODBBackoff([this, ctx]() mutable -> Promise<void> {
			auto t = object -> parent -> conn -> beginTransaction();
			object -> load();
			
			auto f = checkFolder();
			
			auto params = ctx.getParams();
			auto name = params.getName();
			
			// Check if entry name contains a folder
			// If it does, make parent directory and call insert
			// on that.
			KJ_IF_MAYBE(pIdx, name.findLast('/')) {
				auto containingFolder = str(name.slice(0, *pIdx));
				
				auto mkdirRequest = thisCap().mkdirRequest();
				mkdirRequest.setName(containingFolder);
				
				auto newFolder = mkdirRequest.send().getFolder();
				auto tail = newFolder.rmRequest();
				tail.setName(str(name.slice(*pIdx + 1, name.size())));
				
				return ctx.tailCall(mv(tail));
			}
			
			auto entries = f.getEntries();
			
			FolderEntry::Builder entry = nullptr;
			bool entryPresent = false;
			
			auto orphanage = Orphanage::getForMessageContaining(f);
			auto newEntries = orphanage.newOrphan<List<FolderEntry>>(entries.size() - 1);
			
			size_t i = 0;
			for(auto e : entries) {				
				if(e.getName().asString() == name) {
					continue;
				}
				
				// If we are at the last entry and no match, we don't need to modify the folder
				if(i >= newEntries.get().size())
					return READY_NOW;
				
				newEntries.get().setWithCaveats(i++, e);
			}
			
			f.adoptEntries(mv(newEntries));
			
			object -> save();
			return READY_NOW;
		});
	}
	
	Promise<void> mkdir(MkdirContext ctx) override {
		return withODBBackoff([this, ctx] () mutable -> Promise<void> {
			auto t = object -> parent -> conn -> beginTransaction();
			object -> load();
			
			auto f = checkFolder();
			
			auto params = ctx.getParams();
			auto name = params.getName();
			
			// Check if entry name contains a folder
			// If it does, make parent directory and call mkdir
			// on that.
			KJ_IF_MAYBE(pIdx, name.findLast('/')) {
				auto containingFolder = str(name.slice(0, *pIdx));
				
				auto mkdirRequest = thisCap().mkdirRequest();
				mkdirRequest.setName(containingFolder);
				
				auto newFolder = mkdirRequest.send().getFolder();
				auto tail = newFolder.mkdirRequest();
				tail.setName(str(name.slice(*pIdx + 1, name.size())));
				
				return ctx.tailCall(mv(tail));
			}
			
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
			
			if(entryPresent) {
				KJ_REQUIRE(entry.isFolder(), "Non-folder entry already present");
				ctx.getResults().setFolder(entry.getFolder());
			}
			
			// Create new folder object
			int64_t newId = object -> parent -> createObject.insert();
			auto newDBObject = object -> parent -> open(newId);
			newDBObject -> info.initFolder();
			newDBObject -> save();
			
			Folder::Client asFolder = object -> parent -> wrap(mv(newDBObject));
			
			auto orphanage = Orphanage::getForMessageContaining(f);
			auto newList = orphanage.newOrphan<List<FolderEntry>>(1);
		
			auto listRefs = kj::heapArray<List<FolderEntry>::Reader>({newList.get(), f.getEntries()});
			f.adoptEntries(orphanage.newOrphanConcat(listRefs.asPtr()));
			
			entry = f.getEntries()[0];
			entry.setName(name);
			entry.setFolder(asFolder);
			object -> save();
			
			ctx.getResults().setFolder(asFolder);
			return READY_NOW;
		});
	}
};
	
//! Wrapper hook to indicate that a capability is derived from a database object or being exported as such
struct ObjectDB::ObjectHook : public ClientHook, public kj::Refcounted {
	Own<ClientHook> inner;
	inline static const uint BRAND = 0;
	Own<DBObject> object;
			
	//! Waits until this object has settled into a usable state
	Promise<void> whenReady() {
		// The promise in the database could potentially
		// be of an exception the type OVERLOADED (wondering whether that
		// really is smart to do ...), so we need to avoid returning
		// it inside withODBBackoff.
		return withODBBackoff([this]() {
			object -> load(); 
		}).then([this]() -> Promise<void> {
			auto info = object -> info;
			switch(info.which()) {
				case ObjectInfo::UNRESOLVED:
					return object -> whenUpdated().then([this]() { return whenReady(); });
				case ObjectInfo::EXCEPTION: {
					return fromProto(info.getException());
				}	
				case ObjectInfo::LINK:
				case ObjectInfo::DATA_REF:
				case ObjectInfo::FOLDER:
					return READY_NOW;
			}
			
			KJ_FAIL_REQUIRE("Unknown object entry in database");
		});
	}
	
	ObjectHook(Own<DBObject> objectIn) :
		object(mv(objectIn))
	{
		KJ_ASSERT(object.get() != nullptr);
		
		Promise<Own<ClientHook>> innerPromise = whenReady()
		.then([this]() -> Own<ClientHook> {
			// If object is a link, return a client hook pointing to the target
			if(object -> info.isLink()) {
				Object::Client target = object -> info.getLink();
				auto unwrapResult = object -> getParent().unwrap(target);
				
				if(unwrapResult.is<decltype(nullptr)>()) {
					return ClientHook::from(capnp::Capability::Client(nullptr));
				}
				
				if(unwrapResult.is<Own<DBObject>>()) {
					return kj::heap<ObjectHook>(mv(unwrapResult.get<Own<DBObject>>()));
				}
				
				KJ_FAIL_REQUIRE("Invalid link target");
			}
			
			// Otherwise, create an ObjectImpl pointing to the resolved object
			Capability::Client client = kj::heap<ObjectImpl>(object->addRef());
			return ClientHook::from(mv(client));
		});
		
		inner = capnp::newLocalPromiseClient(mv(innerPromise));
	}

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
		return inner -> whenMoreResolved();
		//return nullptr;
	}

	virtual kj::Own<ClientHook> addRef() override { return kj::addRef(*this); }

	const void* getBrand() override { return &BRAND; }

	kj::Maybe<int> getFd() override { return nullptr; }
};

struct ObjectDB::DownloadProcess : public internal::DownloadTask<Own<DBObject>> {
	Own<DBObject> dst;
	Own<BlobBuilder> builder;
	
	DownloadProcess(Own<DBObject> dst, DataRef<AnyPointer>::Client src) :
		DownloadTask(src, Context()),
		dst(mv(dst))
	{}
	
	auto inTransaction(bool readWrite = false) { return dst -> parent -> conn -> beginRootTransaction(readWrite); }
	
	Promise<Maybe<Own<DBObject>>> unwrap() override {
		return withODBBackoff([this]() mutable -> Maybe<Own<DBObject>> {
			KJ_DBG("Unwrapping");
			auto t = inTransaction(false);
			auto unwrapResult = dst -> parent -> unwrap(src);
			
			if(unwrapResult.is<Own<DBObject>>()) {
				dst -> info.setLink(dst -> parent -> wrap(mv(unwrapResult.get<Own<DBObject>>())));
				dst -> save();
				return mv(dst);
			}
			
			return nullptr;
		});
	}
	
	Promise<Maybe<Own<DBObject>>> useCached() override {
		return withODBBackoff([this]() mutable -> Maybe<Own<DBObject>> {
			auto t = inTransaction(true);
			
			// Initialize target to DataRef
			auto ref = dst -> info.initDataRef();
			ref.setMetadata(metadata);
			auto capTableOut = ref.initCapTable(capTable.size());
			for(auto i : kj::indices(capTable))
				capTableOut.set(i, dst -> parent -> download(capTable[i].castAs<DataRef<AnyPointer>>()));
			
			// Check if we have hash in blob store
			KJ_IF_MAYBE(pBlob, dst -> parent -> blobStore -> find(metadata.getDataHash())) {
				ref.getDownloadStatus().setFinished();
				pBlob -> incRef();
				dst -> save();
				return mv(dst);
			}
			
			// Allocate blob builder
			builder = kj::heap<BlobBuilder>(*(dst -> parent -> blobStore));
			
			return nullptr;
		});
	}
	
	Promise<void> receiveData(kj::ArrayPtr<const kj::byte> data) override {
		return withODBBackoff([this, data]() mutable {
			KJ_REQUIRE(builder.get() != nullptr);
			
			// Always grab the write lock before doing anything to fail fast
			auto t = inTransaction(true);
			builder -> write(data);
		});
	}
	
	Promise<void> finishDownload() override {
		return withODBBackoff([this]() mutable {
			KJ_REQUIRE(builder.get() != nullptr);
			
			auto t = inTransaction(true);
			
			auto finishedBlob = builder -> finish();
			finishedBlob.incRef();
			
			dst -> info.getDataRef().getDownloadStatus().setFinished();
			dst -> save();
		});
	}
	
	Promise<Own<DBObject>> buildResult() override {
		return mv(dst);
	}	
};

// ========================= class ObjectDB =============================

namespace {
	struct DownloadErrorHandler : public kj::TaskSet::ErrorHandler {
		void taskFailed(kj::Exception&& exception) override {
			KJ_LOG(WARNING, "Download task failed (should never happen)", exception);
		}
		
		static DownloadErrorHandler instance;
	};
	
	DownloadErrorHandler DownloadErrorHandler::instance;
}

ObjectDB::ObjectDB(kj::StringPtr filename, kj::StringPtr tablePrefix, bool readOnly) :
	filename(str(filename)), tablePrefix(str(tablePrefix)), readOnly(readOnly)
{
	shared = true;
	if(filename == ":memory:" ||filename == "")
		shared = false;
	
	// KJ_LOG(WARNING, "Unshared ObjectDB usage is not tested");
	
	conn = openSQLite3(filename);
	
	auto objectsTable = str(tablePrefix, "_objects");
	auto refsTable = str(tablePrefix, "_object_refs");
	
	if(!readOnly) {
		conn -> exec(str(
			"CREATE TABLE IF NOT EXISTS ", objectsTable, " ("
			  "id INTEGER PRIMARY KEY,"
			  "info BLOB,"
			  "refcount INTEGER DEFAULT 0"
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
		
		insertRef = conn -> prepare(str("INSERT INTO ", refsTable, " (parent, slot, child) VALUES (?, ?, ?)"));
		clearOutgoingRefs = conn -> prepare(str("DELETE FROM ", refsTable, " WHERE parent = ?"));
	}
	
	getInfo = conn -> prepare(str("SELECT info FROM ", objectsTable, " WHERE id = ?"));
	getRefcount = conn -> prepare(str("SELECT refcount FROM ", objectsTable, " WHERE id = ?"));
	
	listOutgoingRefs = conn -> prepare(str("SELECT child FROM ", refsTable, " WHERE parent = ? ORDER BY slot"));
	
	blobStore = kj::refcounted<BlobStore>(*conn, tablePrefix, readOnly);
	
	if(!readOnly)
		createRoot();
}

Own<DBObject> ObjectDB::open(int64_t id) {
	return kj::refcounted<DBObject>(*this, id);
}

void ObjectDB::createRoot() {
	auto t = conn -> beginTransaction();
	
	// Check if object exists already
	auto q = getInfo.query(1);
	if(q.step())
		return;
	
	Temporary<ObjectInfo> oInfo;
	oInfo.initFolder();
	
	auto asBytes = wordsToBytes(messageToFlatArray(oInfo));
	
	auto insertStatement = conn -> prepare(str("INSERT INTO ", tablePrefix, "_objects (id, info, refcount) VALUES (1, ?, 1)"));
	insertStatement.insert(asBytes.asPtr());
}

Folder::Client ObjectDB::getRoot() {
	return wrap(open(1));
}

Object::Client ObjectDB::wrap(Maybe<Own<DBObject>> object) {
	KJ_IF_MAYBE(pObject, object) {
		Own<ClientHook> wrapper = kj::refcounted<ObjectHook>(mv(*pObject));
		return Object::Client(mv(wrapper));
	}
	
	return nullptr;
}


Own<sqlite::Connection> ObjectDB::forkConnection(bool readOnly) {
	if(!shared)
		return conn->addRef();
	
	return openSQLite3(filename);
}

OneOf<Capability::Client, Own<DBObject>, decltype(nullptr)> ObjectDB::unwrap(Capability::Client object) {
	// Resolve to innermost hook
	// If any of the hooks encountered is an object hook, use that
	Own<capnp::ClientHook> hook = capnp::ClientHook::from(kj::cp(object));
	
	// First check whether this is a DB object in our database ...
	Maybe<Own<DBObject>> extractedObject;
	
	ClientHook* inner = hook.get();
	while(true) {
		if(inner -> getBrand() == &ObjectHook::BRAND) {
			auto asObjectHook = static_cast<ObjectHook*>(inner);
			
			if(asObjectHook -> object -> parent.get() == this)
				extractedObject = asObjectHook -> object -> addRef();
		}
		
		KJ_IF_MAYBE(pHook, inner -> getResolved()) {
			inner = pHook;
		} else {
			break;
		}
	}
	
	KJ_IF_MAYBE(pObj, extractedObject) {
		return mv(*pObj);
	}
	
	// ... otherwise check whether it is a null cap ... 
	if(inner -> isNull()) {
		return nullptr;
	}
	
	// ... turns out we can't unwrap this.
	return mv(object);
}

Object::Client ObjectDB::download(DataRef<AnyPointer>::Client object) {
	// Step 1: Check if object is a nullptr or a local DB object
	auto unwrapped = unwrap(object);
	
	if(unwrapped.is<decltype(nullptr)>()) {
		return nullptr;
	}
	
	if(unwrapped.is<Own<DBObject>>()) {
		return wrap(mv(unwrapped.get<Own<DBObject>>()));
	}
	
	// Step 2: Start download task
	auto asCap = unwrapped.get<Capability::Client>();
	Own<DBObject> dbObject = startDownloadTask(asCap.castAs<DataRef<AnyPointer>>());
	return wrap(mv(dbObject));
}

Own<DBObject> ObjectDB::startDownloadTask(DataRef<AnyPointer>::Client object) {	
	// Initialize the object to an unresolved state
	int64_t id = createObject.insert();
	auto dbObject = open(id);
	dbObject -> info.setUnresolved();
	dbObject -> save();
	
	// Create download process
	auto downloadProcess = kj::refcounted<DownloadProcess>(dbObject -> addRef(), object);
	
	auto downloadThenCleanup = canceler.wrap(
			downloadProcess -> output()
			.ignoreResult()
			
			// If download fails, set object to exception
			.catch_([this, dbObject = dbObject -> addRef()](kj::Exception e) mutable {
				return withODBBackoff([this, e, dbObject = mv(dbObject)]() mutable {
					auto t = conn -> beginRootTransaction(true);
					dbObject -> info.setException(toProto(e));
					dbObject -> save();
				});
			})
		)
		// Ignore further exceptions
		.catch_([](kj::Exception&& e) {})
		.then([this, id]() { whenResolved.erase(id); })
		.eagerlyEvaluate(nullptr)
	;
	
	whenResolved.insert(std::make_pair(id, downloadThenCleanup.fork()));
	
	return dbObject;
}

Promise<void> ObjectDB::drain() {
	if(whenResolved.empty())
		return READY_NOW;
	
	kj::Vector<Promise<void>> promises;
	for(auto& idPromisePair : whenResolved) {
		promises.add(idPromisePair.second.addBranch());
	}
	
	return kj::joinPromises(promises.releaseAsArray())
	.then([db = addRef()]() mutable {
		return db -> drain();
	});
}
	
Promise<void> ObjectDB::downloadTask(DataRef<AnyPointer>::Client src, int64_t id) {
	using RemoteRef = DataRef<AnyPointer>;
	using MetaCPResponse = Response<RemoteRef::MetaAndCapTableResults>;
	
	// Wait for both to be downloaded
	return src.metaAndCapTableRequest().send()
	.then([this, src, id](MetaCPResponse response) mutable {
		return withODBBackoff([this, src, id, response = mv(response)]() mutable -> Promise<void> {
			auto t = conn -> beginTransaction();
			
			auto dst = open(id);
			auto refData = dst -> info.initDataRef();
			
			refData.setMetadata(response.getMetadata());
			refData.getMetadata().setDataHash(nullptr);
			
			// Copy cap table over, wrap child objects in their own download processes
			auto capTableIn = response.getTable();
			auto capTableOut = refData.initCapTable(capTableIn.size());
			for(auto i : kj::indices(capTableIn)) {
				capTableOut.set(i, download(capTableIn[i].castAs<DataRef<AnyPointer>>()));
			}
			
			auto dataSize = response.getMetadata().getDataSize();
			
			// Check if the data already exist in the blob db
			{
				KJ_IF_MAYBE(pBlob, blobStore -> find(response.getMetadata().getDataHash())) {
					refData.getDownloadStatus().setFinished();
					pBlob -> incRef();
					dst -> save();
					return READY_NOW;
				}
			}
			
			refData.getDownloadStatus().setDownloading();
				
			auto downloadRequest = src.transmitRequest();
			downloadRequest.setStart(0);
			downloadRequest.setEnd(dataSize);
			downloadRequest.setReceiver(kj::heap<TransmissionReceiver>(*dst));
			
			dst -> save();
			
			return downloadRequest.send().ignoreResult()
			.then([this, id]() mutable {
				return withODBBackoff([this, id]() {
					auto t = conn -> beginTransaction();
					auto dst = open(id);
					
					dst -> load();
					
					auto refData = dst -> info.getDataRef();
					refData.getDownloadStatus().setFinished();
					dst -> save();
				});
			});
		});
	});
}

void ObjectDB::deleteIfOrphan(int64_t id) {
	KJ_REQUIRE(!readOnly);
	
	{
		auto rc = getRefcount.query(id);
		KJ_REQUIRE(rc.step(), "Internal error, refcount not found");
		
		if(rc[0].asInt64() > 0)
			return;
	}
	
	auto dbo = open(id);
	dbo -> load();
	
	kj::ArrayPtr<const byte> hash = nullptr;
	
	if(dbo -> info.isDataRef()) {
		hash = dbo -> info.getDataRef().getMetadata().getDataHash();
	}
	
	if(hash != nullptr) {
		// Decrease refcount on blob
		KJ_IF_MAYBE(pBlob, blobStore -> find(hash)) {
			pBlob -> decRef();
		}
	}
		
	// Scan outgoing refs
	std::set<int64_t> idsToCheck;
	{
		auto outgoingRefs = listOutgoingRefs.query(id);
		while(outgoingRefs.step()) {
			// Skip NULL
			if(outgoingRefs[0].type() == sqlite::Type::NULLTYPE) {
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

// ========================================== class DBObject ================================================

DBObject::DBObject(ObjectDB& parent, int64_t id) :
	info(nullptr), id(id), parent(parent.addRef())
{
	infoHolder = kj::heap<MallocMessageBuilder>();
	info = infoHolder -> getRoot<ObjectInfo>();
}

DBObject::~DBObject() {}

Promise<void> DBObject::whenUpdated() {	
	// Check we have an active download task
	auto pDLT = parent -> whenResolved.find(id);
	if(pDLT != parent -> whenResolved.end()) {
		return pDLT -> second.addBranch();
	}
	
	// Wait a fixed time
	return getActiveThread().timer().afterDelay(5 * kj::SECONDS);
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
			// Skip NULL
			if(outgoingRefs[0].type() == sqlite::Type::NULLTYPE) {
				continue;
			}
			
			auto target = outgoingRefs[0].asInt64();
			parent -> decRefcount(target);
			idsToCheck.insert(target);
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
}

void DBObject::load() {
	// All work here has to be done inside a DB transaction
	auto t = parent -> conn -> beginTransaction();
	
	auto q = parent -> getInfo.query(id);
	KJ_REQUIRE(q.step(), "Object not present in database");
	
	auto flatInfo = kj::heapArray<const byte>(q[0].asBlob());
	auto heapBuffer = bytesToWords(mv(flatInfo));
	FlatArrayMessageReader reader(heapBuffer);
	
	kj::Vector<Maybe<Own<ClientHook>>> rawCapTable;
	
	auto refs = parent -> listOutgoingRefs.query(id);
	while(refs.step()) {
		auto col = refs[0];
		
		if(col.type() == sqlite::Type::NULLTYPE) {
			rawCapTable.add(nullptr);
			continue;
		}
		
		auto dbObject = parent -> open(col);
		rawCapTable.add(ClientHook::from(parent -> wrap(mv(dbObject))));
	}
	
	ReaderCapabilityTable capTable(rawCapTable.releaseAsArray());
	auto root = capTable.imbue(reader.getRoot<ObjectInfo>());
	
	infoHolder = kj::heap<MallocMessageBuilder>(root.totalSize().wordCount + 1);
	infoHolder -> setRoot(root);
	
	info = infoHolder -> getRoot<ObjectInfo>();
}

}}

// =========================== class DBCache =======================

