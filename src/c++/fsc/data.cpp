#include <kj/array.h>
#include <capnp/generated-header-support.h>
#include <capnp/serialize.h>
#include <capnp/orphan.h>
#include <kj/filesystem.h>
#include <kj/map.h>

#include <botan/hash.h>

#include <functional>
#include <cstdlib>

#include <fsc/data-archive.capnp.h>
#include <capnp/rpc.capnp.h>

#include "data.h"
#include "db-cache.h"
#include "sqlite.h"

using capnp::WORDS;
using capnp::word;

namespace fsc {
	
namespace {	
	kj::Exception fromProto(capnp::rpc::Exception::Reader proto) { 
		kj::Exception::Type type;
		switch(proto.getType()) {
			#define HANDLE_VAL(val) \
				case capnp::rpc::Exception::Type::val: \
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
	
	Temporary<capnp::rpc::Exception> toProto(kj::Exception& e) {
		Temporary<capnp::rpc::Exception> result;
		
		switch(e.getType()) {
			#define HANDLE_VAL(val) \
				case kj::Exception::Type::val: \
					result.setType(capnp::rpc::Exception::Type::val); \
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
	
	struct LocalRefServerSet : capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>> {
		using Base = capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>>;
		
		DataRef<capnp::AnyPointer>::Client wrap(Own<internal::LocalDataRefImplV2> s) {
			return add(mv(s));
		}
		
		Promise<Maybe<internal::LocalDataRefImplV2&>> tryUnwrap(DataRef<capnp::AnyPointer>::Client& c) {
			return getLocalServer(c)
			.then([](Maybe<DataRef<capnp::AnyPointer>::Server&> s) -> Maybe<internal::LocalDataRefImplV2&> {
				KJ_IF_MAYBE(pS, s) {
					return static_cast<internal::LocalDataRefImplV2&>(*pS);
				}
				
				return nullptr;
			});
		}
	};
	
	static LocalRefServerSet LOCAL_REF_SERVER_SET = LocalRefServerSet();
	
	size_t getDefaultRamObjectLimit() {
		try {
			char* envValueRaw = getenv("FUSIONSC_RAM_OBJECT_LIMIT");
			
			if(envValueRaw != nullptr) {
				return kj::StringPtr(envValueRaw).parseAs<size_t>();
			}
		} catch(kj::Exception& e) {
			KJ_LOG(WARNING, "Failed to parse FUSIONSC_RAM_OBJECT_LIMIT env variable as size_t, defaulting to 500MB");
		}
		
		return 500000000;
	}
	
	static const size_t RAM_OBJECT_LIMIT = getDefaultRamObjectLimit();
}
	
Promise<bool> isDataRef(capnp::Capability::Client clt) {
	auto asRef = clt.castAs<DataRef<capnp::AnyPointer>>();
	auto req = asRef.metaAndCapTableRequest();
	return req.send().ignoreResult()
	.then(
		// Success case
		[](){
			return true;
		},
		
		// Error case
		[](kj::Exception e) {
			if(e.getType() != kj::Exception::Type::UNIMPLEMENTED)
				throw e;
			
			return false;
		}
	);
}

// === class MMapTemporary ===
kj::Array<byte> internal::MMapTemporary::request(size_t size) {			
	// Otherwise, if the object is big, give it is own file
	auto tmpFile = dir -> createTemporary();
	tmpFile -> truncate(size);
	auto mapping = tmpFile -> mmapWritable(0, size);
	
	auto ptr = mapping->get();
	return ptr.attach(mv(mapping));
};

// === class LocalDataService ===

LocalDataService::LocalDataService(const LibraryHandle& hdl) :
	LocalDataService(*kj::refcounted<internal::LocalDataServiceImpl>(hdl))
{}

LocalDataService::LocalDataService(internal::LocalDataServiceImpl& newImpl) :
	impl(newImpl.addRef())
{}

// Non-const copy constructor
LocalDataService::LocalDataService(LocalDataService& other) : 
	impl(other.impl -> addRef())
{}
// Copy assignment operator
LocalDataService& LocalDataService::operator=(LocalDataService& other) {
	impl = other.impl -> addRef();
	return *this;
}

LocalDataService::operator DataService::Client() {
	return impl -> addRef();
}
	
LocalDataRef<capnp::Data> LocalDataService::publish(kj::ArrayPtr<const byte> bytes) {
	return publish<capnp::Data::Reader>(bytes);
}

LocalDataRef<capnp::Data> LocalDataService::publish(kj::Array<const byte> bytes, kj::ArrayPtr<const byte> hash) {
	Temporary<DataRefMetadata> metaData;
	
	metaData.setId(getActiveThread().randomID());
	metaData.getFormat().setRaw();
	metaData.setCapTableSize(0);
	metaData.setDataSize(bytes.size());
	metaData.setDataHash(hash);
	
	return publish(metaData, mv(bytes));
}

LocalDataRef<capnp::Data> LocalDataService::publishFile(const kj::ReadableFile& file, kj::ArrayPtr<const kj::byte> fileHash, bool copy) {
	kj::Array<const byte> data;
	
	if(copy) {
		data = file.readAllBytes();
	} else {
		auto metadata = file.stat();
		data = file.mmap(0, metadata.size);
	}
	
	return publish(mv(data), fileHash);
}

LocalDataRef<capnp::Data> LocalDataService::publishFile(const kj::ReadableFile& file, bool copy) {
	return publishFile(file, nullptr, copy);
}

void LocalDataService::setLimits(Limits newLimits) { impl->setLimits(newLimits); }

void LocalDataService::setChunkDebugMode() {
	impl -> setChunkDebugMode();
}

// === class internal::LocalDataRefBackend ===

internal::LocalDataRefBackend::LocalDataRefBackend(LocalDataRefGroup& g, StoreEntry e, Temporary<DataRefMetadata>&& metadata, kj::ArrayPtr<capnp::Capability::Client> capTable) :
	group(g),
	storeEntry(mv(e)),
	metadata(mv(metadata))
{
	group.entries.add(*this);
	
	auto ctBuilder = kj::heapArrayBuilder<CapTableEntry>(capTable.size());
	for(auto& e : capTable) {
		ctBuilder.add(processCapTableEntry(e));
	}
	
	this -> capTable = ctBuilder.finish();
	data = storeEntry.asArray();
}

internal::LocalDataRefBackend::~LocalDataRefBackend() {
	if(groupLink.isLinked())
		group.entries.remove(*this);
}

Own<internal::LocalDataRefBackend> internal::LocalDataRefBackend::deepFork(LocalDataRefGroup& group) {
	KJ_REQUIRE(capTable.size() == 0, "Can only fork DataRefs with empty capability table");
	
	Temporary<DataRefMetadata> newMd(metadata.asReader());
	
	return kj::refcounted<LocalDataRefBackend>(group, storeEntry, mv(newMd), nullptr);
}

Array<capnp::Capability::Client> internal::LocalDataRefBackend::getCapTable() {
	KJ_REQUIRE(groupLink.isLinked(), "Internal error: getCapTable() called from non-external reference");

	auto builder = kj::heapArrayBuilder<capnp::Capability::Client>(capTable.size());
	
	for(CapTableEntry& e : capTable) {
		capnp::Capability::Client forked = e.addBranch()
		.then([g = group.addRef()](CapTableData oneOf) -> capnp::Capability::Client {
			if(oneOf.is<capnp::Capability::Client>())
				return oneOf.get<capnp::Capability::Client>();
			
			auto pBackend = oneOf.get<Shared<LocalDataRefBackend>>();
			return LOCAL_REF_SERVER_SET.wrap(
				kj::refcounted<LocalDataRefImplV2>(*pBackend)
			);
		});
		builder.add(forked);
	}
	
	return builder.finish();
}

internal::LocalDataRefBackend::CapTableEntry internal::LocalDataRefBackend::processCapTableEntry(capnp::Capability::Client c) {
	DataRef<capnp::AnyPointer>::Client asRef = c.castAs<DataRef<capnp::AnyPointer>>();
	
	return LOCAL_REF_SERVER_SET.tryUnwrap(asRef)
	.then([c, &myGroup = this -> group](Maybe<LocalDataRefImplV2&> maybeRef) mutable -> CapTableData {
		KJ_IF_MAYBE(pRef, maybeRef) {
			auto& backend = *(pRef -> backend);
			
			// Check if backend is from same group.
			// Otherwise we need to treat it as if external ref
			if(&(backend.group) != &myGroup)
				return mv(c);
				
			// We need to keep the client alive until after this
			// promise resolves. If we do not, the containing
			// promise might get destroyed during the deallocation
			// of the surrounding lambda, because the client holds
			// a reference to the group that the internal reference
			// does not hold.
			auto keepAlive = kj::evalLater([c](){});
			getActiveThread().detach(mv(keepAlive));
			
			return Shared<LocalDataRefBackend>(backend.addRefInternal());
		}
		
		return mv(c);
	})
	.fork();
}

Own<internal::LocalDataRefBackend> internal::LocalDataRefBackend::addRefExternal() {
	KJ_REQUIRE(groupLink.isLinked(), "Internal error: Adding external ref after group is destroyed");
	return addRefInternal().attach(group.addRef());
}

Array<const byte> internal::LocalDataRefBackend::forkData() {
	return storeEntry.asArray();
}

// === class internal::LocalDataRefGroup ===

internal::LocalDataRefGroup::~LocalDataRefGroup() {
	// We need to keep all contents temporarily alive
	// so that the contents list doesn't change
	auto keepAlive = kj::heapArrayBuilder<Own<LocalDataRefBackend>>(entries.size());
	for(auto& e : entries)
		keepAlive.add(e.addRefInternal());
	
	// Clear all capability tables
	// This breaks all recursive links of objects with each
	// other, so that everything can get cleared up in time.
	for(auto& e : entries) {
		e.capTable = nullptr;
		entries.remove(e);
	}
	
	keepAlive.clear();
}

// === class internal::LocalDataRefImplV2 ===

internal::LocalDataRefImplV2::LocalDataRefImplV2(LocalDataRefBackend& b) :
	backend(b.addRefExternal())
{}

Own<internal::LocalDataRefImplV2> internal::LocalDataRefImplV2::deepFork() {
	auto newGroup = kj::refcounted<LocalDataRefGroup>();
	auto newBackend = backend -> deepFork(*newGroup);
	return kj::refcounted<LocalDataRefImplV2>(*newBackend).attach(mv(newGroup));
}

kj::ArrayPtr<capnp::Capability::Client> internal::LocalDataRefImplV2::getCapTable() {
	KJ_IF_MAYBE(pTbl, cachedCapTable) {
		return *pTbl;
	}
	
	return cachedCapTable.emplace(backend -> getCapTable());
}

capnp::AnyPointer::Reader internal::LocalDataRefImplV2::getRoot(const capnp::ReaderOptions& opts) {
	KJ_REQUIRE(!getMetadata().getFormat().isRaw(), "Can not obtain message root in raw data");
	
	capnp::ReaderCapabilityTable* rct;
	capnp::FlatArrayMessageReader* mr;
	
	// If neccessary, construct reader capability table
	KJ_IF_MAYBE(pTbl, this -> readerCapTable) {
		rct = pTbl;
	} else {
		auto capTable = getCapTable();
		auto rctBuilder = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(capTable.size());
		
		for(auto client : capTable) {
			auto hook = capnp::ClientHook::from(mv(client));
			if(hook -> isNull()) {
				rctBuilder.add(nullptr);
			} else {
				rctBuilder.add(mv(hook));
			}
		}
		
		rct = &(this -> readerCapTable.emplace(rctBuilder.finish()));
	}
	
	KJ_IF_MAYBE(pReader, this -> messageReader) {
		mr = pReader;
	} else {
		// Obtain data as a byte pointer (note that this drops all attached objects to keep alive0
		ArrayPtr<const byte> bytePtr = getRaw();
		
		// Cast the data to a word array (let's hope they are aligned properly)
		ArrayPtr<const capnp::word> wordPtr = ArrayPtr<const capnp::word>(
			reinterpret_cast<const capnp::word*>(bytePtr.begin()),
			bytePtr.size() / sizeof(capnp::word)
		);
		
		mr = &(messageReader.emplace(wordPtr, opts));
	}
	
	return rct -> imbue(mr -> getRoot<capnp::AnyPointer>());
}

Promise<void> internal::LocalDataRefImplV2::metaAndCapTable(MetaAndCapTableContext context) {	
	context.getResults().setMetadata(getMetadata());
	
	auto tblIn = getCapTable();
	auto tblOut = context.getResults().initTable(tblIn.size());
	for(auto i : kj::indices(tblIn))
		tblOut.set(i, tblIn[i]);
	
	return kj::READY_NOW;
}

Promise<void> internal::LocalDataRefImplV2::rawBytes(RawBytesContext context) {
	uint64_t start = context.getParams().getStart();
	uint64_t end   = context.getParams().getEnd();
	
	auto ptr = getRaw();
	
	KJ_REQUIRE(end >= start);
	KJ_REQUIRE(start < ptr.size());
	KJ_REQUIRE(end <= ptr.size());
	
	if(end == start)
		return kj::READY_NOW;
	
	context.getResults().setData(ptr.slice(start, end));
	
	return kj::READY_NOW;
}

namespace {	
	struct TransmissionProcess {
		constexpr static inline size_t CHUNK_SIZE = 1024 * 1024;
		
		DataRef<capnp::AnyPointer>::Receiver::Client receiver;
		size_t end;
		
		Array<const byte> data;
		
		TransmissionProcess() :
			receiver(nullptr)
		{}
		
		Promise<void> run(size_t start) {
			KJ_REQUIRE(end >= start);
			
			auto request = receiver.beginRequest();
			request.setNumBytes(end - start);
			return request.send().ignoreResult().then([this, start]() { return transmit(start); });
		}
		
		Promise<void> transmit(size_t start) {
			size_t chunkEnd = start + CHUNK_SIZE;
			
			if(chunkEnd > end)
				chunkEnd = end;
			
			// Check if we are done transmitting
			if(chunkEnd == start)
				return receiver.doneRequest().send().ignoreResult();
			
			auto slice = data.slice(start, chunkEnd);
			
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
			
			return request.send().then([this, chunkEnd]() { return transmit(chunkEnd); });
		}
	};
}

Promise<void> internal::LocalDataRefImplV2::transmit(TransmitContext context) {
	auto params = context.getParams();
	
	auto process = kj::heap<TransmissionProcess>();
	
	// Prepare a pointer to the data that will also keep the data alive
	process -> data = getRaw().attach(addRef());
	process -> end = params.getEnd();
	process -> receiver = params.getReceiver();
	
	auto result = process -> run(params.getStart());
	
	return result.attach(mv(process));
}	

// === class internal::LocalDataServiceImpl ===

namespace {
	Own<DBCache> createSqliteTempCache() {
		auto conn = connectSqlite("");
		auto blobStore = createBlobStore(*conn);
		return createDBCache(*blobStore);
	}
}

internal::LocalDataServiceImpl::LocalDataServiceImpl(const LibraryHandle& hdl) :
	backingStore(hdl.store()),
	fileBackedMemory(kj::newDiskFilesystem()->getCurrent().clone()),
	dbCache(createSqliteTempCache())
{
	limits.maxRAMObjectSize = RAM_OBJECT_LIMIT;
}

Own<internal::LocalDataServiceImpl> internal::LocalDataServiceImpl::addRef() {
	return kj::addRef(*this);
}

void internal::LocalDataServiceImpl::setLimits(LocalDataService::Limits newLimits) {
	limits = newLimits;
}

kj::Array<byte> internal::LocalDataServiceImpl::allocate(size_t size) {
	if(size <= limits.maxRAMObjectSize)
		return kj::heapArray<byte>(size);
	else
		return fileBackedMemory.request(size);
}

void internal::LocalDataServiceImpl::setChunkDebugMode() {
	debugChunks = true;
}

Promise<Maybe<LocalDataRef<capnp::AnyPointer>>> internal::LocalDataServiceImpl::unwrap(DataRef<capnp::AnyPointer>::Client src) {
	return LOCAL_REF_SERVER_SET.getLocalServer(src).then([this, src](Maybe<DataRef<capnp::AnyPointer>::Server&> maybeServer) mutable -> Maybe<LocalDataRef<capnp::AnyPointer>> {
		auto unwrap = [src](DataRef<capnp::AnyPointer>::Server& srv) mutable {
			auto& backend = static_cast<internal::LocalDataRefImplV2&>(srv);
			return LocalDataRef<capnp::AnyPointer>(src, backend.addRef());
		};
		
		return maybeServer.map(unwrap);
	}).attach(addRef());
}

struct internal::LocalDataServiceImpl::DataRefDownloadProcess : public DownloadTask<LocalDataRef<capnp::AnyPointer>> {
	Own<LocalDataServiceImpl> service;
	bool recursive;
	
	StoreEntry dataEntry = nullptr;
	
	kj::Array<kj::byte> downloadBuffer;
	size_t downloadOffset = 0;
	
	DataRefDownloadProcess(LocalDataServiceImpl& service, DataRef<capnp::AnyPointer>::Client src, bool recursive, DTContext ctx) :
		DownloadTask(mv(src), mv(ctx)), service(service.addRef()), recursive(recursive)
	{}
	
	Promise<Maybe<ResultType>> unwrap() override {
		if(recursive)
			return Maybe<ResultType>(nullptr);
		
		return service -> unwrap(src);
	}
	
	capnp::Capability::Client adjustRef(capnp::Capability::Client ref) override {
		if(!recursive)
			return mv(ref);
		
		capnp::Capability::Client adjusted = service -> download(ref.castAs<DataRef<capnp::AnyPointer>>(), true, this -> ctx);
		
		return adjusted.whenResolved()
		.then([adjusted]() mutable { return adjusted;  })
		.catch_([ref](kj::Exception&& e) mutable {				
			if(e.getType() == kj::Exception::Type::UNIMPLEMENTED)
				return ref;
			throw e;
		});
	}
	
	virtual Promise<Maybe<ResultType>> useCached() override {
		auto dataHash = metadata.getDataHash();
		if(dataHash.size() > 0) {			
			KJ_IF_MAYBE(rowPtr, service -> backingStore.query(dataHash)) {
				dataEntry = mv(*rowPtr);
				return buildResult()
				.then([](ResultType result) {
					return Maybe<ResultType>(mv(result));
				});
			}
		}
		
		return Maybe<ResultType>(nullptr);
	}
	
	Promise<void> beginDownload() override {
		downloadBuffer = service -> allocate(metadata.getDataSize());
		downloadOffset = 0;
		return READY_NOW;
	}
	
	Promise<void> receiveData(kj::ArrayPtr<const kj::byte> data) override {
		// KJ_DBG(data.slice(0, 12));
		KJ_REQUIRE(downloadOffset + data.size() <= downloadBuffer.size());
		memcpy(downloadBuffer.begin() + downloadOffset, data.begin(), data.size());
		
		downloadOffset += data.size();
		return READY_NOW;
	}
	
	Promise<void> finishDownload() override {
		KJ_REQUIRE(downloadOffset == downloadBuffer.size());
		KJ_REQUIRE(downloadBuffer != nullptr);
		
		/*uint32_t* prefix = reinterpret_cast<uint32_t*>(downloadBuffer.begin());
		uint32_t nSegments = prefix[0] + 1;
		
		kj::ArrayPtr<uint32_t> segmentSizes(prefix + 1, nSegments);
		
		size_t expected = nSegments / 2 + 1;
		for(auto s : segmentSizes)
			expected += s;
		
		KJ_DBG("Finalizing downloaded ref", nSegments, segmentSizes, expected, downloadOffset / 8);*/
		
		// Note: The hash is computed in the parent class.
		dataEntry = service -> backingStore.publish(metadata.getDataHash(), mv(downloadBuffer));
		
		return READY_NOW;
	}
	
	Promise<ResultType> buildResult() override {
		LocalDataRefGroup& group = *(ctx.localGroup);
		
		auto backend = kj::refcounted<internal::LocalDataRefBackend>(
			group,
			mv(dataEntry),
			mv(metadata),
			capTable
		);
		auto impl = kj::refcounted<internal::LocalDataRefImplV2>(*backend);
		
		if(recursive)
			return ResultType(impl -> addRef(), impl -> addRef());
		else
			return ResultType(this -> src, mv(impl));
	}
};

Promise<LocalDataRef<capnp::AnyPointer>> internal::LocalDataServiceImpl::download(DataRef<capnp::AnyPointer>::Client src, bool recursive, DTContext ctx) {
	return unwrap(src).then([this, src, recursive, ctx = mv(ctx)](Maybe<LocalDataRef<capnp::AnyPointer>> maybeUnwrapped) mutable -> Promise<LocalDataRef<capnp::AnyPointer>> {
		KJ_IF_MAYBE(pUnwrapped, maybeUnwrapped) {
			return *pUnwrapped;
		}
		
		auto downloadProcess = kj::refcounted<DataRefDownloadProcess>(*this, mv(src), recursive, mv(ctx));
		return downloadProcess -> output();
	});
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publish(DataRefMetadata::Reader metaData, Array<const byte>&& data, ArrayPtr<capnp::Capability::Client> capTable, Maybe<LocalDataRefGroup&> maybeGroup) {	
	KJ_REQUIRE(data.size() >= metaData.getDataSize(), "Data do not fit inside provided array");
	KJ_REQUIRE(metaData.getCapTableSize() == capTable.size(), "Specified capability count must match provided table");
	
	Temporary<DataRefMetadata> myMetaData = metaData;
	
	// Check if hash is nullptr. If yes, construct own.
	if(myMetaData.getDataHash().size() == 0) {
		auto hashFunction = getActiveThread().library() -> defaultHash();
		hashFunction -> update(data.begin(), data.size());
		
		KJ_STACK_ARRAY(uint8_t, hashOutput, hashFunction -> output_length(), 1, 64);
		hashFunction -> final(hashOutput.begin());
		
		myMetaData.setDataHash(hashOutput);
	}
	
	auto storeEntry = backingStore.publish(myMetaData.getDataHash(), mv(data));
	
	Own<LocalDataRefGroup> lg;
	KJ_IF_MAYBE(pGroup, maybeGroup) {
		lg = pGroup -> addRef();
	} else {
		lg = kj::refcounted<LocalDataRefGroup>();
	}
	
	auto backend = kj::refcounted<LocalDataRefBackend>(
		*lg,
		mv(storeEntry),
		mv(myMetaData),
		capTable
	);
	
	auto impl = kj::refcounted<LocalDataRefImplV2>(*backend);
	return LocalDataRef<capnp::AnyPointer>(impl -> addRef(), impl -> addRef());
}

Promise<void> internal::LocalDataServiceImpl::hash(HashContext ctx) {
	using DR = DataRef<capnp::AnyPointer>;
	
	auto ref = ctx.getParams().getSource();
	return ref.metaAndCapTableRequest().send()
	.then([this, ctx, ref = mv(ref)](capnp::Response<DR::MetaAndCapTableResults> response) mutable {
		auto hashFunction = getActiveThread().library() -> defaultHash();
		
		auto asBytes = wordsToBytes(capnp::canonicalize(response.getMetadata()));
		hashFunction -> update(asBytes.begin(), asBytes.size());
		
		Promise<void> hashAll = READY_NOW;
		
		for(auto child : response.getTable()) {
			hashAll = hashAll.then([this, child = mv(child)]() mutable {
				auto childHashReq = asClient().hashRequest<capnp::AnyPointer>();
				childHashReq.setSource(mv(child).castAs<DataRef<capnp::AnyPointer>>());
				return childHashReq.send();
			})
			.then([&hf = *hashFunction](auto response) mutable {
				hf.update("Success");
				auto hash = response.getHash();
				hf.update(hash.begin(), hash.size());
			}).catch_([&hf = *hashFunction](kj::Exception e) mutable {
				hf.update("Failure");
			});
		}
		
		return hashAll.then([ctx, &hf = *hashFunction]() mutable {
			KJ_STACK_ARRAY(uint8_t, hashOutput, hf.output_length(), 1, 64);
			hf.final(hashOutput.begin());
			
			ctx.getResults().setHash(hashOutput);
		}).attach(mv(hashFunction));
	});
}

/*

 Documentation of archive format:
 
 Nomenclature:
   WORD : Elementary size unit, corresponds to 8 bytes
   <u64: Unsigned 64-bit integer
 
 Format structure:
 
 - Magic tag (8 bytes): ASCII encoding of the 7 characters FSCARCH followed by a null terminator byte
 - Header size (8 bytes, <u64): Size of the header in WORDs, must be at least 3
 
 - Header (N WORDs as given by Header size):
   - Description size (8 bytes, <u64): Size of text description in WORDs
   - Data size (8 bytes, <u64): Size of data section in WORDs
   - Info size (8 bytes, <u64): Size of info section in WORDs
   - Remaining bytes to reach word count specified in Header size
  
 - Text description (N WORDs as given by description size):
    This section contains an arbitrary text designed to be readable as an info message if this file were
    to be opened by a text editor. Do not interpret the contents of this section in any way. Note that despite
    the contained string incl. null-terminator not neccessarily sharing this property, the size of this section is
    always a multiple of 8 bytes (hence the size info in WORDs) to keep the later sections aligned to 8-byte boundaries
    (which is required by CapnProto).

 - Data section (N WORDs as given by data section size):
    This section contains all the contents of BLOBs used to back DataRefs. It is placed in front of the info section
	to allow streaming downloads of data (at that time the size of the info section is not known).
	
 - Info section (N WORDs as given by info section size):
    This section contains a message holding a fsc::ArchiveInfo struct as its root, serialized using Capn'n'proto'safe
	flat array serialization format.
	
*/
 

struct ArchiveWriter {
	
	static inline kj::StringPtr MAGIC_TAG = "FSCARCH"_kj;
	static inline kj::StringPtr DESCRIPTION = "This is an FSC / fusionsc archive file. To read it, please use the fusionsc toolkit to inspect its contents or refer it for details on the format"_kj;
	
	static inline capnp::WordCount MAGIC_TAG_SIZE = 1 * WORDS;
	static inline capnp::WordCount HEADER_SIZE_SIZE = 1 * WORDS;
	static inline capnp::WordCount HEADER_SIZE = 3 * WORDS;
	static inline capnp::WordCount DESCRIPTION_SIZE = (DESCRIPTION.size() + 7) / 8 * WORDS;
	static inline capnp::WordCount TOTAL_PREFIX_SIZE = MAGIC_TAG_SIZE + HEADER_SIZE_SIZE + HEADER_SIZE + DESCRIPTION_SIZE;
	
	struct DataRecord {
		uint64_t id;
		capnp::WordCount offsetWords;
		uint64_t sizeBytes;
		// Own<const kj::WritableFileMapping> mapping;
		uint64_t globalOffset;
		kj::ListLink<DataRecord> link;
	};
	
	struct InfoRecord {
		uint64_t id;
		Temporary<ArchiveInfo::ObjectInfo> info;
		kj::ListLink<InfoRecord> link;
	};
	
	const kj::File& file;
	
	//! Current size of data section
	capnp::WordCount dataSize = 0 * WORDS;
	
	//! Promise that resolves when all DataRefs known to advertise (don't trust this) a given hash have finished or failed downloading
	kj::TreeMap<ID, Promise<void>> downloadQueue;
	
	//! Map of all completed data blocks by hash
	kj::TreeMap<ID, uint64_t> dataRecordsByHash;
	
	//! List of all data records
	kj::List<DataRecord, &DataRecord::link> dataRecords;
	
	//! List of all object info records
	kj::List<InfoRecord, &InfoRecord::link> infoRecords;
	
	//! Download context to use for de-duplication
	internal::DownloadTask<uint64_t>::Context downloadContext;
	
	ArchiveWriter(const kj::File& file) :
		file(file)
	{}
	
	~ArchiveWriter() {
		for(auto& record : dataRecords) {
			dataRecords.remove(record);
			delete &record;
		}
		
		for(auto& record : infoRecords) {
			infoRecords.remove(record);
			delete &record;
		}
	}
	
	void write(capnp::word* dst, uint64_t value) {
		auto* wVal = reinterpret_cast<capnp::_::WireValue<uint64_t>*>(dst);
		wVal -> set(value);
	}
	
	void writePrefix() {
		file.truncate(TOTAL_PREFIX_SIZE / WORDS * sizeof(word));
		
		auto mapping = file.mmapWritable(0, TOTAL_PREFIX_SIZE / WORDS * sizeof(word));
		word* const mappingStart = reinterpret_cast<word*>(mapping -> get().begin());
		word* buf = mappingStart;
		
		memcpy(buf, MAGIC_TAG.begin(), MAGIC_TAG_SIZE / WORDS * sizeof(word));
		buf += MAGIC_TAG_SIZE / WORDS;
		
		write(buf, HEADER_SIZE / WORDS);
		buf += HEADER_SIZE_SIZE / WORDS;
		
		write(buf, DESCRIPTION_SIZE / WORDS);
		buf += 1;
		
		write(buf, 0); // Data size will be written later in writeInfo
		buf += 1;
		
		write(buf, 0); // Info size will be written later in writeInfo
		buf += 1;
		
		memset(buf, 0, DESCRIPTION_SIZE / WORDS * sizeof(word));
		memcpy(buf, DESCRIPTION.begin(), DESCRIPTION.size());
		buf += DESCRIPTION_SIZE;
				
		mapping -> sync(mapping -> get());
	}
	
	//! Allocates data from the data section and creats a data record
	DataRecord& allocData(uint64_t nBytes) {
		capnp::WordCount nWords = (nBytes + 7) / 8 * WORDS;
		KJ_ASSERT(sizeof(word) * nWords / WORDS >= nBytes);
		
		file.truncate((TOTAL_PREFIX_SIZE + dataSize + nWords) / WORDS * sizeof(word));
		
		auto* bl = new DataRecord;
		bl -> id = dataRecords.size();
		bl -> offsetWords = dataSize;
		bl -> sizeBytes = nBytes;
		// bl -> mapping = file.mmapWritable((TOTAL_PREFIX_SIZE + dataSize) / WORDS * sizeof(word), nWords / WORDS * sizeof(word));
		bl -> globalOffset = (TOTAL_PREFIX_SIZE + dataSize) / WORDS * sizeof(word);
		dataRecords.add(*bl);
		
		dataSize += nWords;
		return *bl;
	}
	
	//! Allocates a new info record
	InfoRecord& allocInfo() {
		auto* rec = new InfoRecord;
		rec -> id = infoRecords.size();
		infoRecords.add(*rec);
		
		return *rec;
	}
	
	//! Serializes the info section and finalizes the header 
	void writeInfo(Temporary<ArchiveInfo> infoSection) {
		kj::Array<word> flatMessage = messageToFlatArray(*(infoSection.holder));
		
		file.truncate((TOTAL_PREFIX_SIZE + dataSize + flatMessage.size() * WORDS) / WORDS * sizeof(word));
		
		// Write info section to disk
		{
			auto mapping = file.mmapWritable((TOTAL_PREFIX_SIZE + dataSize) / WORDS * sizeof(word), flatMessage.size() * sizeof(word));
			memcpy(mapping -> get().begin(), flatMessage.begin(), flatMessage.size() * sizeof(word));
			
			mapping -> sync(mapping -> get());
		}
		
		// Write data and info size into header
		{
			auto mapping = file.mmapWritable((MAGIC_TAG_SIZE + HEADER_SIZE_SIZE + 1 /* Skip description */) / WORDS * sizeof(word), 2 * sizeof(word));
			word* buf = reinterpret_cast<word*>(mapping -> get().begin());
			write(buf, dataSize / WORDS);
			write(buf + 1, flatMessage.size());
			mapping -> sync(mapping -> get());
		}
	}
	
	//! Finalizes the file by writing root object and storing the file
	void finalize(uint64_t rootObject) {
		Temporary<ArchiveInfo> infoSection;
		
		// Store all data records
		auto dataInfo = infoSection.initData(dataRecords.size());
		for(auto& dataRecord : dataRecords) {
			auto out = dataInfo[dataRecord.id];
			out.setOffsetWords(dataRecord.offsetWords);
			out.setSizeBytes(dataRecord.sizeBytes);
		}
		
		auto objectInfo = infoSection.initObjects(infoRecords.size());
		for(auto& infoRecord : infoRecords) {
			objectInfo.setWithCaveats(infoRecord.id, infoRecord.info);
		}
		
		infoSection.setRoot(rootObject);
				
		writeInfo(mv(infoSection));
		
		// Synchronize file to disk
		file.sync();
	}
	
	Promise<OneOf<std::nullptr_t, kj::Exception, uint64_t>> downloadRef(capnp::Capability::Client src) {
		auto hook = capnp::ClientHook::from(cp(src));
		if(hook -> isNull())
			return nullptr;
		
		auto transProc = kj::refcounted<TransmissionProcess>(*this, src.castAs<DataRef<capnp::AnyPointer>>());
		
		return transProc -> output()
		.then([](uint64_t result) -> OneOf<std::nullptr_t, kj::Exception, uint64_t> {
			return result;
		})
		.catch_([](kj::Exception e) -> OneOf<std::nullptr_t, kj::Exception, uint64_t> {
			if(e.getType() == kj::Exception::Type::UNIMPLEMENTED)
				return nullptr;
			return mv(e);
		});
	}
	
	Promise<void> writeArchive(DataRef<capnp::AnyPointer>::Client src) {
		writePrefix();
		
		auto transProc = kj::refcounted<TransmissionProcess>(*this, src);
		
		return transProc -> output()
		.then([this](uint64_t result) {
			finalize(result);
		});
	}
	
	struct TransmissionProcess : public internal::DownloadTask<uint64_t> {
		ArchiveWriter& parent;
		
		Maybe<DataRecord&> block;
		size_t writeOffset = 0;
		
		TransmissionProcess(ArchiveWriter& parent, DataRef<capnp::AnyPointer>::Client src) :
			DownloadTask<uint64_t>(src, parent.downloadContext),
			parent(parent)
		{}
		
		// unwrap() not overridden
		// adjustRef() not overridden, recursive downloads are started during finalization
		
		Promise<Maybe<uint64_t>> useCached() override {
			ID key = metadata.getDataHash().asConst();
			
			Promise<void> prereq = READY_NOW;
			
			// If we have active downloads for this key, let them finish first
			KJ_IF_MAYBE(pResult, parent.downloadQueue.find(key)) {
				prereq = mv(*pResult);
				*pResult = output().ignoreResult().catch_([](kj::Exception e) {});
			}
			
			return prereq.then([this, key = mv(key)]() mutable -> Promise<Maybe<uint64_t>> {
				KJ_IF_MAYBE(pResult, parent.dataRecordsByHash.find(key)) {
					// The data are already in the file. Just reference the same block again.
					return finalize(*pResult)
					.then([](uint64_t x) -> Maybe<uint64_t> {
						return x;
					});
				}
				
				auto dataHash = metadata.getDataHash().asConst();
				
				// Get a ref to the store entry if it is found
				Maybe<StoreEntry> maybeStoreEntry = nullptr;
				if(dataHash.size() > 0) {
					maybeStoreEntry = getActiveThread().library() -> store().query(dataHash);
				}
				
				KJ_IF_MAYBE(pStoreEntry, maybeStoreEntry) {
					// We have the block locally. Just allocate space for it and memcpy
					// it over					
					auto data = pStoreEntry -> asPtr();
					metadata.setDataSize(data.size());
					
					auto& dataRecord = parent.allocData(data.size());
					/*kj::byte* target = dataRecord.mapping -> get().begin();
					
					memcpy(target, data.begin(), data.size());
					dataRecord.mapping -> sync(dataRecord.mapping -> get());*/
					parent.file.write(dataRecord.globalOffset, data);
					
					// KJ_DBG("Stored local block", dataRecord.id, dataRecord.mapping -> get(), data);
					
					return finalize(dataRecord.id)
					.then([](uint64_t x) -> Maybe<uint64_t> {
						return x;
					});
				}
				
				return Maybe<uint64_t>(nullptr);
			});
		}
		
		Promise<void> beginDownload() override { 
			block = parent.allocData(metadata.getDataSize());
			return READY_NOW;
		}
		
		Promise<void> receiveData(kj::ArrayPtr<const kj::byte> data) override {
			KJ_IF_MAYBE(pBlock, block) {
				KJ_REQUIRE(writeOffset + data.size() <= pBlock -> sizeBytes, "Overflow in download");
				/*kj::ArrayPtr<kj::byte> output = pBlock -> mapping -> get();
				memcpy(output.begin() + writeOffset, data.begin(), data.size());*/
				parent.file.write(pBlock -> globalOffset + writeOffset, data);
				writeOffset += data.size();
				
				return READY_NOW;
			} else {
				KJ_FAIL_REQUIRE("Failed to allocate data");
			}
		}
		
		Promise<void> finishDownload() override { return READY_NOW; }
		
		Promise<uint64_t> buildResult() override {
			KJ_IF_MAYBE(pBlock, block) {
				// kj::ArrayPtr<kj::byte> output = pBlock -> mapping -> get();
				if(writeOffset < /*output.size()*/pBlock -> sizeBytes) {
					// memset(output.begin() + writeOffset, 0, output.size() - writeOffset);
					parent.file.zero(pBlock -> globalOffset + writeOffset, /*output.size()*/pBlock -> sizeBytes - writeOffset);
				}
				
				// pBlock -> mapping -> sync(output);
				parent.file.sync();
				
				parent.dataRecordsByHash.insert(ID(metadata.getDataHash()), pBlock -> id);
				
				return finalize(pBlock -> id);
			} else {
				KJ_FAIL_REQUIRE("Failed to allocate data");
			}
		}
		
		Promise<uint64_t> finalize(uint64_t blockID) {
			InfoRecord& newRecord = parent.allocInfo();
			
			newRecord.info.setMetadata(metadata);
			newRecord.info.setDataId(blockID);
			
			kj::Vector<Promise<void>> childDownloads;
			
			auto refs = newRecord.info.initRefs(capTable.size());
			for(auto i : kj::indices(capTable)) {
				auto& client = capTable[i];
				auto out = refs[i];
				
				Promise<void> processRef = parent.downloadRef(client)
				.then([this, out](OneOf<std::nullptr_t, kj::Exception, uint64_t> downloadResult) mutable {
					if(downloadResult.is<std::nullptr_t>()) {
						out.setNull();
					} else if(downloadResult.is<kj::Exception>()) {
						out.setException(toProto(downloadResult.get<kj::Exception>()));
					} else if(downloadResult.is<uint64_t>()) {
						out.setObject(downloadResult.get<uint64_t>());
					}
				});
				
				childDownloads.add(mv(processRef));
			}
			
			return kj::joinPromises(childDownloads.releaseAsArray())
			.then([id = newRecord.id]() {
				return id;
			});
		}
	};
};

namespace {
	struct SharedArrayHolder : public kj::AtomicRefcounted {
		kj::Array<const capnp::word> data;
	};
	
	struct FileMappable : public internal::LocalDataServiceImpl::Mappable {
		const kj::ReadableFile& f;
		
		FileMappable(const kj::ReadableFile& f) : f(f) {} 
		kj::Array<const kj::byte> mmap(size_t start, size_t size) override {
			return f.mmap(start, size);
		}
	};
	
	struct ArrayMappable : public internal::LocalDataServiceImpl::Mappable, public kj::AtomicRefcounted {
		kj::Array<const kj::byte> backend;
		
		ArrayMappable(kj::Array<const kj::byte> backend) :
			backend(mv(backend))
		{
			// Make sure that we are allocated through atomicRefcounted
			(void) kj::atomicAddRef(*this);
		}
		
		kj::Array<const kj::byte> mmap(size_t start, size_t size) override {
			return backend.slice(start, start + size).attach(kj::atomicAddRef(*this));
		}
	};
	
	struct ConstantMappable : public internal::LocalDataServiceImpl::Mappable {
		kj::ArrayPtr<const kj::byte> backend;
		
		ConstantMappable(kj::ArrayPtr<const kj::byte> backend) :
			backend(mv(backend))
		{}
		
		kj::Array<const kj::byte> mmap(size_t start, size_t size) override {
			return backend.slice(start, start + size).attach();
		}
	};
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publishArchive(const kj::ReadableFile& f, const capnp::ReaderOptions options) {
	FileMappable fm(f);
	return publishArchive(fm, mv(options));
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publishArchive(kj::Array<const kj::byte> array, const capnp::ReaderOptions options) {
	auto am = kj::atomicRefcounted<ArrayMappable>(mv(array));
	return publishArchive(*am, mv(options));
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publishConstant(kj::ArrayPtr<const kj::byte> array, const capnp::ReaderOptions options) {
	ConstantMappable cm(array);
	return publishArchive(cm, mv(options));
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publishArchive(Mappable& f, const capnp::ReaderOptions options) {
	auto read = [](const word* ptr) {
		return reinterpret_cast<const capnp::_::WireValue<uint64_t>*>(ptr) -> get();
	};
	
	Array<const kj::byte> prefixMapping = f.mmap(0, 5 * sizeof(word));
	const word* prefixStart = reinterpret_cast<const word*>(prefixMapping.begin());
	
	// Check magic tag
	KJ_REQUIRE(prefixMapping.slice(0, 8) == ArchiveWriter::MAGIC_TAG.slice(0, 8).asBytes(), "Invalid magic tag");
	
	// Reader headers
	capnp::WordCount headerSize = read(prefixStart + 1) * WORDS;
	KJ_REQUIRE(headerSize / WORDS >= 3, "Could not load archive file, header size is too small");
	capnp::WordCount descSize = read(prefixStart + 2);
	capnp::WordCount dataSize = read(prefixStart + 3);
	capnp::WordCount infoSize = read(prefixStart + 4);
	
	// Calculate section offsets
	capnp::WordCount dataOffset = 2 * WORDS + headerSize + descSize;
	capnp::WordCount infoOffset = dataOffset + dataSize;
	
	// Load file info
	Array<const kj::byte> infoSection = f.mmap(infoOffset / WORDS * sizeof(word), infoSize / WORDS * sizeof(word));
	capnp::FlatArrayMessageReader infoReader(bytesToWords(infoSection.asPtr()));
	
	ArchiveInfo::Reader archiveInfo = infoReader.getRoot<ArchiveInfo>();
	
	auto objectInfo = archiveInfo.getObjects();
	auto dataInfo = archiveInfo.getData();
	
	// Create mappings from promises to fulfillers for ref dependencies
	kj::Vector<ForkedPromise<LocalDataRef<capnp::AnyPointer>>> refPromises(objectInfo.size());
	kj::Vector<Own<PromiseFulfiller<LocalDataRef<capnp::AnyPointer>>>> refFulfillers(objectInfo.size());
	for(auto i : kj::indices(objectInfo)) {
		auto paf = kj::newPromiseAndFulfiller<LocalDataRef<capnp::AnyPointer>>();
		refPromises.add(paf.promise.fork());
		refFulfillers.add(mv(paf.fulfiller));
	}
	
	Maybe<LocalDataRef<capnp::AnyPointer>> root;
	
	auto group = kj::refcounted<LocalDataRefGroup>();
	
	// Scan all refs, looking to feed out root and fulfill any dependencies
	for(auto i : kj::indices(objectInfo)) {
		auto object = objectInfo[i];
		auto refs = object.getRefs();
		
		auto refBuilder = kj::heapArrayBuilder<capnp::Capability::Client>(refs.size());
		for(auto iRef : kj::indices(refs)) {
			auto refInfo = refs[iRef];
			if(refInfo.isNull()) {
				refBuilder.add(nullptr);
			} else if(refInfo.isException()) {
				refBuilder.add(fromProto(refInfo.getException()));
			} else if(refInfo.isObject()) {
				refBuilder.add(refPromises[refInfo.getObject()].addBranch());
			} else {
				refBuilder.add(KJ_EXCEPTION(DISCONNECTED, "Unknown object type"));
			}
		}
		
		auto dataRecord = dataInfo[object.getDataId()];
		
		kj::Array<const kj::byte> dataMapping = f.mmap((dataOffset / WORDS + dataRecord.getOffsetWords()) * sizeof(word), dataRecord.getSizeBytes());
		
		LocalDataRef<capnp::AnyPointer> published = publish(
			object.getMetadata(),
			mv(dataMapping),
			refBuilder.finish(),
			*group
		);
		
		if(i == archiveInfo.getRoot())
			root = published;
		
		refFulfillers[i] -> fulfill(mv(published));
	}
	
	KJ_IF_MAYBE(pRoot, root) {
		if(pRoot -> getFormat().isUnknown()) {
			KJ_LOG(WARNING, "The root node of the archive has an unspecified type. This will make loading the archive difficult on dynamically typed languages");
		}
		
		return mv(*pRoot);
	}
	
	KJ_FAIL_REQUIRE("Root object could not be located in archive", archiveInfo.getRoot(), archiveInfo.getObjects().size());
}


Promise<void> internal::LocalDataServiceImpl::writeArchive(DataRef<capnp::AnyPointer>::Client ref, const kj::File& out) {
	Shared<ArchiveWriter> writer(out);
	
	return writer -> writeArchive(mv(ref)).attach(kj::cp(writer));
}


Promise<void> internal::LocalDataServiceImpl::clone(CloneContext context) {
	context.getResults().setRef(dbCache -> cache(context.getParams().getSource()));
	return READY_NOW;
}

Promise<void> internal::LocalDataServiceImpl::cloneAllIntoMemory(CloneAllIntoMemoryContext ctx) {
	return download(ctx.getParams().getSource(), true)
	.then([this, ctx](auto localRef) mutable {
		ctx.initResults().setRef(localRef);
	});
}

Promise<void> internal::LocalDataServiceImpl::store(StoreContext context) {
	using capnp::AnyPointer;
	using capnp::AnyList;
	using capnp::ElementSize;
	
	auto params = context.getParams();
	
	AnyPointer::Reader inData = params.getData();
	// uint64_t typeId = params.getStructType();
	auto schema = params.getSchema();
	
	Array<byte> data = nullptr;
	
	capnp::BuilderCapabilityTable capTable;
	
	if(schema.isNull()) {
		// typeId == 0 indicates raw byte data
		
		// Check format
		KJ_REQUIRE(inData.isList());
		auto asList = inData.getAs<AnyList>();
		
		KJ_REQUIRE(asList.getElementSize() == ElementSize::BYTE);
		
		// Copy into memory
		capnp::Data::Reader inAsData = inData.getAs<capnp::Data>();
		
		data = allocate(inAsData.size());
		memcpy(data.begin(), inAsData.begin(), inAsData.size());
	} else {
		// typeID != 0 indicates struct or capability
		
		// Check format
		// Only structs and capabilities can be typed
		// Lists have no corresponding schema nodes (and therefore no IDs)
		KJ_REQUIRE(inData.isStruct() || inData.isCapability());
		
		capnp::MallocMessageBuilder mb;
		AnyPointer::Builder root = capTable.imbue(mb.initRoot<AnyPointer>());
		root.set(inData);
		
		size_t dataSize = capnp::computeSerializedSizeInWords(mb) * sizeof(capnp::word);
		data = allocate(dataSize);
		
		kj::ArrayOutputStream os(data);
		capnp::writeMessage(os, mb);
	}
		
	Temporary<DataRefMetadata> metaData;
	metaData.setId(params.getId());
	
	if(schema.isNull())
		metaData.getFormat().setRaw();
	else
		metaData.getFormat().initSchema().set(schema);
	metaData.setCapTableSize(capTable.getTable().size());
	metaData.setDataSize(data.size());
	
	auto rawTbl = capTable.getTable();
	auto clientBuilder = kj::heapArrayBuilder<capnp::Capability::Client>(rawTbl.size());
	for(auto& maybeHook : rawTbl) {
		KJ_IF_MAYBE(pHook, maybeHook) {
			clientBuilder.add(mv(*pHook));
		} else {
			clientBuilder.add(nullptr);
		}
	}

	auto ref = publish(
		metaData,
		mv(data),
		clientBuilder.finish()
	);
	
	auto cachedRef = dbCache -> cache(mv(ref));
	context.getResults().setRef(mv(cachedRef));
	
	return READY_NOW;
}

// === Flat file storage ===

namespace {

struct FileWriter : public DataRef<capnp::Data>::Receiver::Server {
	Own<const kj::File> out;
	uint64_t offset = 0;
	
	FileWriter(Own<const kj::File>&& newOut) : out(mv(newOut)) {}
	
	Promise<void> begin(BeginContext ctx) {
		out -> truncate(0);
		
		return READY_NOW;
	}
	
	Promise<void> receive(ReceiveContext ctx) {
		auto data = ctx.getParams().getData();
		out -> write(offset, data);
		offset += data.size();
		
		return READY_NOW;
	}
	
	Promise<void> done(DoneContext ctx) {
		out -> sync();
		return READY_NOW;
	}
};

}

Promise<void> LocalDataService::downloadIntoFile(DataRef<capnp::Data>::Client clt, Own<const kj::File>&& dst) {
	auto transmitRequest = clt.transmitRequest();
	transmitRequest.setReceiver(kj::heap<FileWriter>(mv(dst)));
	return transmitRequest.send().ignoreResult();
}

// === function attachToClient ===

namespace internal {

namespace {

struct Proxy : public capnp::Capability::Server {
	capnp::Capability::Client backend;
	
	using capnp::Capability::Server::DispatchCallResult;
	
	inline Proxy(capnp::Capability::Client newBackend) :
		backend(mv(newBackend))
	{}
	
	DispatchCallResult dispatchCall(
		uint64_t interfaceId,
		uint16_t methodId,
        capnp::CallContext<capnp::AnyPointer, capnp::AnyPointer> context
	) {
		auto request = backend.typelessRequest(interfaceId, methodId, context.getParams().targetSize(), capnp::Capability::Client::CallHints());
		request.set(context.getParams());
		context.releaseParams();
		
		return { context.tailCall(mv(request)).attach(thisCap()), false, true };
	}
};

}

kj::Own<capnp::Capability::Server> createProxy(capnp::Capability::Client client) {
	return kj::heap<Proxy>(mv(client));
}

}

// === function hasMaximumOrdinal ===

struct OrdinalChecker {
	capnp::DynamicStruct::Reader in;
	unsigned int maxOrdinal;
	
	Array<byte> mask;
	Array<bool> ptrMask;
	
	using DynamicValue = capnp::DynamicValue;
	using DynamicStruct = capnp::DynamicStruct;
	using StructSchema = capnp::StructSchema;
	using AnyStruct = capnp::AnyStruct;
	using SType = capnp::schema::Type;
	
	OrdinalChecker(DynamicStruct::Reader in, unsigned int maxOrdinal) :
		in(in),
		maxOrdinal(maxOrdinal),
		mask(kj::heapArray<byte>(((AnyStruct::Reader) in).getDataSection().size())),
		ptrMask(kj::heapArray<bool>(((AnyStruct::Reader) in).getPointerSection().size()))
	{
		for(unsigned int i = 0; i < mask.size(); ++i)
			mask[i] = 0;
		
		for(unsigned int i = 0; i < ptrMask.size(); ++i)
			ptrMask[i] = false;
	}
	
	void allowPointer(unsigned int offset) {
		if(offset > ptrMask.size())
			return;
		
		ptrMask[offset] = true;
	}
	
	void allowBool(unsigned int offset) {
		unsigned int byteOffset = offset / 8;
		unsigned int offsetInByte = offset % 8;
		
		if(byteOffset > mask.size())
			return;
		
		mask[byteOffset] |= ((byte) 1) << offsetInByte;
	}
	
	void allowBytes(unsigned int size, unsigned int offsetInSizes) {
		unsigned int byteStart = size * offsetInSizes;
		unsigned int byteEnd   = size * (offsetInSizes + 1);
		
		for(unsigned int i = byteStart; i < byteEnd && i < mask.size(); ++i)
			mask[i] = ~((byte) 0);
	}
	
	bool checkField(DynamicStruct::Reader group, StructSchema::Field field, bool forbidden) {
		// Check if field has ordinal and ordinal is too large
		auto proto = field.getProto();
		// KJ_LOG(WARNING, "Checking field ", proto.getName());
		
		auto ordinal = proto.getOrdinal();
		capnp::DynamicValue::Reader value = group.get(field);
		
		if(ordinal.isExplicit() && ordinal.getExplicit() > maxOrdinal) {
			// KJ_LOG(WARNING, "Ordinal out of range, switching to forbidden mode", ordinal, maxOrdinal);
			forbidden = true;
		}
		
		// Check if field is struct field
		if(proto.isGroup()) {
			KJ_ASSERT(value.getType() == DynamicValue::STRUCT);
			return checkStruct(value.as<DynamicStruct>(), forbidden);			
		}
		
		// If capnproto has new field types and they are used here
		if(!proto.isSlot()) {
			// KJ_LOG(WARNING, "Encountered unknown field type (not slot or group)");
			return false;
		}
		
		auto slot = proto.getSlot();
		
		unsigned int offset = slot.getOffset();
		
		if(slot.getType().isVoid() && forbidden)
			return false;
		
		if(forbidden) {
			// KJ_LOG(WARNING, "Ignored forbidden field", offset);
			return true;
		}
		
		// Check field typedClient
		switch(slot.getType().which()) {
			case SType::BOOL:
				allowBool(offset);
				break;
			
			case SType::VOID:				
				break;
			
			case SType::TEXT:
			case SType::DATA:
			case SType::LIST:
			case SType::STRUCT:
			case SType::INTERFACE:
			case SType::ANY_POINTER:
				allowPointer(offset);
				break;
			
			case SType::INT8:
			case SType::UINT8:
				allowBytes(1, offset);
				break;
			
			case SType::INT16:
			case SType::UINT16:
			case SType::ENUM:
				allowBytes(2, offset);
				break;
			
			case SType::INT32:
			case SType::UINT32:
			case SType::FLOAT32:
				allowBytes(4, offset);
				break;
			
			case SType::INT64:
			case SType::UINT64:
			case SType::FLOAT64:
				allowBytes(8, offset);
				break;
			
			default:
				// KJ_LOG(WARNING, "Encountered unknown primitive field type", slot.getType());
				return false;
		}
		
		// KJ_LOG(WARNING, mask);
		return true;
	}
	
	bool checkStruct(capnp::DynamicStruct::Reader in, bool forbidden) {
		StructSchema schema = in.getSchema();
		
		KJ_IF_MAYBE(field, in.which()) {
			// Check union field
			if(!checkField(in, *field, forbidden))
				return false;
		} else {
			// No union can be read, check if there should be union
			auto numUnions = schema.getUnionFields().size();
			KJ_ASSERT(numUnions != 1);
			
			if(numUnions > 0) {
				// Discriminant is set to unknown value
				// KJ_LOG(WARNING, "Encountered unknown discriminant value in struct ", schema.getProto().getDisplayName());
				return false;
			}	
		}
		
		// Now check non-union fields
		auto nonUnion = schema.getNonUnionFields();
		for(auto field : nonUnion) {
			if(!checkField(in, field, forbidden))
				return false;
		}
		
		// Also add discriminant to allowed fields
		if(!forbidden) {
			auto proto = schema.getProto();
			KJ_ASSERT(proto.isStruct());
			
			auto group = proto.getStruct();
			auto discCount = group.getDiscriminantCount();
			KJ_ASSERT(discCount != 1);
			if(discCount > 0) {
				allowBytes(2, group.getDiscriminantOffset());
				// KJ_LOG(WARNING, "Allowing discriminant at ", 2 * group.getDiscriminantOffset());
				// KJ_LOG(WARNING, mask);
			}
		}
		
		return true;
	}
	
	bool checkRoot() {
		return checkStruct(in, false);
	}
	
	bool checkMask() {		
		AnyStruct::Reader anyStruct = in;
		auto dataSection = anyStruct.getDataSection();
		auto pointerSection = anyStruct.getPointerSection();
		
		KJ_ASSERT(mask.size() == dataSection.size());
		KJ_ASSERT(ptrMask.size() == pointerSection.size());
		
		for(unsigned int i = 0; i < mask.size(); ++i) {
			if((dataSection[i] | mask[i]) != mask[i]) {
				// KJ_LOG(WARNING, "Encountered forbidden bits", i);
				// KJ_LOG(WARNING, mask[i]);
				// KJ_LOG(WARNING, dataSection[i]);
				// KJ_LOG(WARNING, mask[i] | dataSection[i]);
				return false;
			}
		}
		
		for(unsigned int i = 0; i < ptrMask.size(); ++i) {
			if(!ptrMask[i] && !pointerSection[i].isNull()) {
				// KJ_LOG(WARNING, "Encountered forbidden pointer", i);
				return false;
			}
		}
		
		// KJ_LOG(WARNING, "Checks passed");
		// KJ_LOG(WARNING, dataSection);
		// KJ_LOG(WARNING, mask);
		
		return true;
	}
	
	bool check() {
		if(!checkRoot())
			return false;
		
		return checkMask();
	}
};

bool hasMaximumOrdinal(capnp::DynamicStruct::Reader in, unsigned int maxOrdinal) {
	return OrdinalChecker(in, maxOrdinal).check();
}

// === function removeDatarefs ===

Promise<void> removeDatarefsInStruct(capnp::AnyStruct::Reader in, capnp::AnyStruct::Builder out);
Promise<void> removeCapability(capnp::Capability::Client client, capnp::AnyPointer::Builder out);

Promise<void> removeDatarefs(capnp::AnyPointer::Reader in, capnp::AnyPointer::Builder out) {
	using capnp::PointerType;
	using capnp::AnyPointer;
	
		
	switch(in.getPointerType()) {
		case PointerType::NULL_: {
			out.clear();
			return READY_NOW;
		}
		case PointerType::LIST: {
			using capnp::AnyList;
			using capnp::AnyStruct;
			using capnp::ElementSize;
			using capnp::List;
			
			// Retrieve as list
			auto anyList = in.getAs<AnyList>();
			auto inSize = anyList.size();
			
			// Check size format
			switch(anyList.getElementSize()) {
				// Case 1: Pointers
				case ElementSize::POINTER: {
					auto pointerListIn  = anyList.as<List<AnyPointer>>();
					auto pointerListOut = out.initAsAnyList(ElementSize::POINTER, inSize).as<List<AnyPointer>>();
					
					auto promises = kj::heapArrayBuilder<Promise<void>>(inSize);
					for(decltype(inSize) i = 0; i < inSize; ++i)
						promises.add(removeDatarefs(pointerListIn[i], pointerListOut[i]));
					
					return joinPromises(promises.finish());
				}
				
				// Case 2: Structs
				case ElementSize::INLINE_COMPOSITE: {
					// Special case: Size 0
					if(inSize == 0) {
						out.initAsListOfAnyStruct(0, 0, 0);
						return READY_NOW;
					}
					
					auto structListIn  = anyList.as<List<AnyStruct>>();
					auto dataWords = structListIn[0].getDataSection().size() / sizeof(capnp::word);
					auto pointerSize = structListIn[0].getPointerSection().size();
					
					auto structListOut = out.initAsListOfAnyStruct(dataWords, pointerSize, inSize);
					
					auto promises = kj::heapArrayBuilder<Promise<void>>(inSize);
					for(decltype(inSize) i = 0; i < inSize; ++i)
						promises.add(removeDatarefsInStruct(structListIn[i], structListOut[i]));
					
					return joinPromises(promises.finish());
				}
				
				// Other sizes are simple data sections, so we can just copy (phew)
				default:
					out.set(in);
			}
			return READY_NOW;
		}
		case PointerType::STRUCT: {
			using capnp::AnyStruct;
			
			auto structIn = in.getAs<AnyStruct>();
			auto dataWords = structIn.getDataSection().size() / sizeof(capnp::word);
			auto pointerSize = structIn.getPointerSection().size();
			return removeDatarefsInStruct(structIn, out.initAsAnyStruct(dataWords, pointerSize));
		}
		case PointerType::CAPABILITY: {
			return removeCapability(in.getAs<capnp::Capability>(), out);
		}
	};
	
	return READY_NOW;
}

Promise<void> removeDatarefs(capnp::AnyStruct::Reader in, capnp::AnyStruct::Builder out) {
	auto result = removeDatarefsInStruct(in, out);
	return result;
}

Promise<void> removeDatarefsInStruct(capnp::AnyStruct::Reader in, capnp::AnyStruct::Builder out) {	
	// Copy over data section
	KJ_REQUIRE(out.getDataSection().size() == in.getDataSection().size());
	memcpy(out.getDataSection().begin(), in.getDataSection().begin(), in.getDataSection().size());
	
	// Handle pointers
	auto nPointers = in.getPointerSection().size();
	auto promises = kj::heapArrayBuilder<Promise<void>>(nPointers);
	
	for(decltype(nPointers) i = 0; i < nPointers; ++i) {
		promises.add(removeDatarefs(in.getPointerSection()[i], out.getPointerSection()[i]));
	}
	
	return joinPromises(promises.finish());
}

Promise<void> removeCapability(capnp::Capability::Client client, capnp::AnyPointer::Builder out) {
	using capnp::AnyPointer;
	
	auto typedClient = client.castAs<DataRef<AnyPointer>>();
	
	return typedClient.metaAndCapTableRequest().send()
	.then([out](auto response) mutable {
		out.setAs<capnp::Data>(response.getMetadata().getId());
	})
	.catch_([out](kj::Exception e) mutable {
		out.clear();
		
		// UNIMPLEMENTED exceptions are normal here, as we want to ignore non-dataref capabilities
		if(e.getType() != kj::Exception::Type::UNIMPLEMENTED)
			kj::throwRecoverableException(mv(e));
		
	});
}

size_t linearIndex(const capnp::List<uint64_t>::Reader& shape, const ArrayPtr<size_t> index) {
	size_t linearIndex = 0;
	size_t stride = 1;
	for(int dim = (int) index.size() - 1; dim >= 0; --dim) {
		linearIndex += index[dim] * stride;
		stride *= shape[dim];
	}
	
	return linearIndex;
}

// DataRef flattening

namespace internal { namespace {

struct FlatData {
	struct RefInfo {
		using Link = OneOf<std::nullptr_t, kj::Exception, uint64_t>;
		
		Temporary<DataRefMetadata> metadata;
		kj::Vector<Link> links;
		uint64_t data;
	};
	
	kj::Vector<RefInfo> refs;
	kj::Vector<kj::Array<const byte>> data;
	
	// Caching data that helps with downloading
	std::unordered_map<capnp::ClientHook*, uint64_t> storedRefs;
	std::unordered_map<const byte*, uint64_t> storedData;
	
	Own<LocalDataServiceImpl> lds;
	
	FlatData(LocalDataServiceImpl& lds) :
		lds(lds.addRef())
	{}
	
	uint64_t storeRefData(LocalDataRef<capnp::AnyPointer> ref) {
		const byte* key = ref.getRaw().begin();
		auto it = storedData.find(key);
		if(it != storedData.end()) {
			return it -> second;
		}
		
		uint64_t idx = data.size();
		storedData.insert(std::make_pair(key, idx));
		data.add(ref.forkRaw());
		
		return idx;
	}
	
	Promise<uint64_t> processRef(DataRef<capnp::AnyPointer>::Client ref) {
		auto hook = capnp::ClientHook::from(DataRef<capnp::AnyPointer>::Client(ref));
		
		auto it = storedRefs.find(hook.get());
		if(it != storedRefs.end()) {
			return it -> second;
		}
		
		return lds -> download(ref, false)
		.then([this, hook = mv(hook)](auto ref) mutable {
			return processLocalRef(hook.get(), ref);
		});
	}
	
	Promise<uint64_t> processLocalRef(capnp::ClientHook* original, LocalDataRef<capnp::AnyPointer> ref) {
		uint64_t idx = storedRefs.size();
		storedRefs.insert(std::make_pair(original, idx));
		
		RefInfo& info = refs.add();
		info.metadata = ref.getMetadata();
		info.data = storeRefData(ref);
		
		auto capTable = ref.getCapTable();
		info.links.reserve(capTable.size());
		
		auto downloadPromises = kj::heapArrayBuilder<Promise<void>>(capTable.size());
		for(auto iLink : kj::indices(capTable)) {
			auto entry = capTable[iLink];
			auto& link = info.links.add(nullptr);
			
			auto hook = capnp::ClientHook::from(cp(entry));
			if(hook -> isNull()) {
				downloadPromises.add(READY_NOW);
				continue;
			}
			
			DataRef<>::Client target(hook -> addRef());
			
			Promise<void> downloadPromise = processRef(target)
			.then([this, &link](uint64_t id) {
				link = id;
			})
			.catch_([this, &link](kj::Exception e) {
				link = e;
			});
			downloadPromises.add(mv(downloadPromise));
		}
		
		return kj::joinPromises(downloadPromises.finish())
		.then([idx]() { return idx; });
	}
};

}

Promise<kj::Array<kj::Array<const byte>>> LocalDataServiceImpl::downloadFlat(DataRef<>::Client src) {
	Shared<FlatData> data(*this);
	
	// Step 1: Download root
	return data -> processRef(mv(src))
	
	// Step 2: Flatten in-memory representation
	.then([data](uint64_t rootId) mutable {
		Temporary<ArchiveInfo> graphRepr;
		graphRepr.setRoot(rootId);
		
		auto objects = graphRepr.initObjects(data -> refs.size());
		for(auto i : kj::indices(objects)) {
			auto& in = data -> refs[i];
			auto out = objects[i];
			
			out.setMetadata(in.metadata);
			out.setDataId(in.data);
			
			auto& linksIn = in.links;
			auto linksOut = out.initRefs(linksIn.size());
			
			for(auto iLink : kj::indices(linksIn)) {
				auto& linkIn = linksIn[iLink];
				auto linkOut = linksOut[iLink];
			
				if(linkIn.is<std::nullptr_t>()) {
					linkOut.setNull();
				} else if(linkIn.is<kj::Exception>()) {
					linkOut.setException(toProto(linkIn.get<kj::Exception>()));
				} else {
					linkOut.setObject(linkIn.get<uint64_t>());
				}
			}
		}
		
		auto result = kj::heapArrayBuilder<kj::Array<const byte>>(data -> data.size() + 1);
		
		capnp::BuilderCapabilityTable tmpCapTable; // Unused anyway
		result.add(buildData<ArchiveInfo>(graphRepr, tmpCapTable));
		
		for(auto& arr : data -> data)
			result.add(mv(arr));
		
		return result.finish();
	});
}

LocalDataRef<> LocalDataServiceImpl::publishFlat(kj::Array<kj::Array<const byte>> data) {
	// We need to know whether we reused an array for another ref already, since we need to then steal from that
	using MaybePublished = kj::OneOf<kj::Array<const byte>, LocalDataRef<>>;
	
	// Decode data
	KJ_REQUIRE(data.size() >= 1);
	auto asWords = bytesToWords(data[0].asPtr());
	capnp::FlatArrayMessageReader treeReader(asWords);
	auto info = treeReader.getRoot<ArchiveInfo>();
	
	// Prepare store of maybe published arrays
	auto maybePublished = kj::heapArrayBuilder<MaybePublished>(data.size() - 1);
	for(auto& a : data.slice(1, data.size())) {
		maybePublished.add(mv(a));
	}
	
	// Process all stored refs
	uint64_t nRefs = info.getObjects().size();
	auto refs = kj::heapArrayBuilder<ForkedPromise<LocalDataRef<>>>(nRefs);
	auto refFulfillers = kj::heapArrayBuilder<Own<kj::PromiseFulfiller<LocalDataRef<>>>>(nRefs);
	for(auto i : kj::range(0, nRefs)) {
		auto paf = kj::newPromiseAndFulfiller<LocalDataRef<>>();
		refs.add(paf.promise.fork());
		refFulfillers.add(mv(paf.fulfiller));
	}
	
	Maybe<LocalDataRef<>> result;
	
	auto group = kj::refcounted<LocalDataRefGroup>();
	
	// Fill in references
	for(auto i : kj::range(0, nRefs)) {
		auto refInfo = info.getObjects()[i];
		auto links = refInfo.getRefs();
		
		auto capTableBuilder = kj::heapArrayBuilder<capnp::Capability::Client>(links.size());
		for(auto iLink : kj::indices(links)) {
			auto link = links[iLink];
			
			if(link.isNull()) {
				capTableBuilder.add(nullptr);
				continue;
			}
			
			DataRef<>::Client target(nullptr);
			if(link.isException()) {
				capTableBuilder.add(fromProto(link.getException()));
			} else if(link.isObject()) {
				KJ_REQUIRE(link.getObject() < refs.size(), "Out of bounds link");
				capTableBuilder.add(refs[link.getObject()].addBranch());
			} else {
				KJ_FAIL_REQUIRE("Unknown link type encountered in flat representation");
			}
		}
		
		auto dataId = refInfo.getDataId();
		
		auto& mp = maybePublished[dataId];
		kj::Array<const byte> refData(nullptr);
		
		if(mp.is<kj::Array<const byte>>()) {
			refData = mv(mp.get<kj::Array<const byte>>());
		} else {
			refData = mp.get<LocalDataRef<>>().forkRaw();
		}
		
		auto publishedRef = publish(refInfo.getMetadata(), mv(refData), capTableBuilder.finish(), *group);
		if(mp.is<kj::Array<const byte>>()) {
			mp = publishedRef;
		}
		
		if(i == info.getRoot())
			result = publishedRef;
		
		refFulfillers[i] -> fulfill(mv(publishedRef));
	}
	
	KJ_IF_MAYBE(pResult, result) {
		return mv(*pResult);
	} else {
		KJ_FAIL_REQUIRE("The root ref could not be found in the published refs.");
	}
}

} // namespace internal

// function overrideRefs

namespace {
	struct CapTableOverride : public DataRef<capnp::AnyPointer>::Server {
		DataRef<capnp::AnyPointer>::Client delegate;
		kj::Array<capnp::Capability::Client> capTable;
		
		CapTableOverride(DataRef<capnp::AnyPointer>::Client d, kj::Array<capnp::Capability::Client> t) :
			delegate(mv(d)), capTable(mv(t))
		{}
		
		DispatchCallResult dispatchCall(
			uint64_t interfaceId, uint16_t methodId,
			capnp::CallContext<capnp::AnyPointer, capnp::AnyPointer> context
		) override {
			// Check for metaAndCapTable
			if(interfaceId == capnp::typeId<DataRef<capnp::AnyPointer>>() && methodId == 0) {
				return DataRef<capnp::AnyPointer>::Server::dispatchCall(interfaceId, methodId, mv(context));
			}
			
			auto params = context.getParams();
			auto targetRequest = delegate.typelessRequest(interfaceId, methodId, capnp::MessageSize{ params.targetSize().wordCount + 1 }, capnp::Capability::Client::CallHints());
			targetRequest.set(params);
			
			return DispatchCallResult { context.tailCall(mv(targetRequest)), false, true };
		}
		
		Promise<void> metaAndCapTable(MetaAndCapTableContext ctx) {
			return delegate.metaAndCapTableRequest().send()
			.then([ctx, this](auto response) mutable {
				ctx.setResults(response);
				
				auto tbl = ctx.getResults().initTable(capTable.size());
				for(auto i : kj::indices(capTable))
					tbl.set(i, capTable[i]);
			});
		}
	};
}

DataRef<capnp::AnyPointer>::Client overrideRefs(DataRef<capnp::AnyPointer>::Client base, kj::Array<capnp::Capability::Client> refs) {
	return kj::heap<CapTableOverride>(mv(base), mv(refs));
}

} // namespace fsc
