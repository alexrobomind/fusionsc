#include <kj/array.h>
#include <capnp/generated-header-support.h>
#include <capnp/serialize.h>
#include <capnp/orphan.h>
#include <kj/filesystem.h>
#include <kj/map.h>

#include <botan/hash.h>

#include <functional>

#include <fsc/data-archive.capnp.h>
#include <capnp/rpc.capnp.h>

#include "data.h"
#include "odb.h"

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
	
// === functions in internal ===

template<>
capnp::Data::Reader internal::getDataRefAs<capnp::Data>(internal::LocalDataRefImpl& impl, const capnp::ReaderOptions& options) {
	(void) options;
	return impl.entryRef -> value.asPtr();
}
	
// === class LocalDataService ===

LocalDataService::LocalDataService(Library& lib) :
	LocalDataService(*kj::refcounted<internal::LocalDataServiceImpl>(lib))
{}

LocalDataService::LocalDataService(internal::LocalDataServiceImpl& newImpl) :
	Client(newImpl.addRef()),
	impl(newImpl.addRef())
{}

// Non-const copy constructor
LocalDataService::LocalDataService(LocalDataService& other) : 
	capnp::Capability::Client(other),
	impl(other.impl -> addRef())
{}

// Move constructor
LocalDataService::LocalDataService(LocalDataService&& other) : 
	capnp::Capability::Client(other),
	impl(other.impl -> addRef())
{}

// Copy assignment operator
LocalDataService& LocalDataService::operator=(LocalDataService& other) {
	capnp::Capability::Client::operator=(other);
	impl = other.impl -> addRef();
	return *this;
}

// Copy assignment operator
LocalDataService& LocalDataService::operator=(LocalDataService&& other) {
	capnp::Capability::Client::operator=(other);
	impl = other.impl -> addRef();
	return *this;
}
	
LocalDataRef<capnp::Data> LocalDataService::publish(kj::ArrayPtr<const byte> bytes) {
	return publish<capnp::Data::Reader>(bytes);
}

LocalDataRef<capnp::Data> LocalDataService::publishFile(const kj::ReadableFile& file, kj::ArrayPtr<const kj::byte> fileHash) {
	auto fileData = file.stat();
	kj::Array<const kj::byte> data = file.mmap(0, fileData.size);
	
	Temporary<DataRefMetadata> metaData;
	metaData.setId(getActiveThread().randomID());
	metaData.getFormat().setUnknown();
	metaData.setCapTableSize(0);
	metaData.setDataSize(data.size());
	metaData.setDataHash(fileHash);
	
	return publish(metaData, mv(data));
}

void LocalDataService::setLimits(Limits newLimits) { impl->setLimits(newLimits); }

void LocalDataService::setChunkDebugMode() {
	impl -> setChunkDebugMode();
}

// === class internal::LocalDataRefImpl ===

Own<internal::LocalDataRefImpl> internal::LocalDataRefImpl::addRef() {
	return kj::addRef(*this);
}

DataRefMetadata::Reader internal::LocalDataRefImpl::localMetadata() {
	return _metadata.getRoot<DataRefMetadata>();
}

Promise<void> internal::LocalDataRefImpl::metaAndCapTable(MetaAndCapTableContext context) {
	using CC = capnp::Capability::Client;
	
	context.getResults().setMetadata(localMetadata());
	
	auto results = context.getResults();
	results.initTable((unsigned int) capTableClients.size());
	
	for(unsigned int i = 0; i < capTableClients.size(); ++i) {
		results.getTable().set(i, capTableClients[i]);
	}
	
	return kj::READY_NOW;
}

Promise<void> internal::LocalDataRefImpl::rawBytes(RawBytesContext context) {
	uint64_t start = context.getParams().getStart();
	uint64_t end   = context.getParams().getEnd();
	
	auto ptr = get<capnp::Data>(READ_UNLIMITED);
	
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

Promise<void> internal::LocalDataRefImpl::transmit(TransmitContext context) {
	auto params = context.getParams();
	
	auto process = kj::heap<TransmissionProcess>();
	
	// Prepare a pointer to the data that will also keep the data alive
	process -> data = getDataRefAs<capnp::Data>(*this, READ_UNLIMITED).attach(addRef());
	process -> end = params.getEnd();
	process -> receiver = params.getReceiver();
	
	auto result = process -> run(params.getStart());
	
	return result.attach(mv(process));
}	

capnp::FlatArrayMessageReader& internal::LocalDataRefImpl::ensureReader(const capnp::ReaderOptions& options) {
	KJ_IF_MAYBE(reader, maybeReader) {
		return *reader;
	}
	
	// Obtain data as a byte pointer (note that this drops all attached objects to keep alive0
	ArrayPtr<const byte> bytePtr = getDataRefAs<capnp::Data>(*this, options);
	
	// Cast the data to a word array (let's hope they are aligned properly)
	ArrayPtr<const capnp::word> wordPtr = ArrayPtr<const capnp::word>(
		reinterpret_cast<const capnp::word*>(bytePtr.begin()),
		bytePtr.size() / sizeof(capnp::word)
	);
	
	return maybeReader.emplace(wordPtr, options);
}

kj::Array<const byte> internal::LocalDataRefImpl::addRefRaw() {
	const LocalDataStore::Entry& entry = *entryRef;
	return entry.value.asPtr().attach(entry.addRef());
}

// === class internal::LocalDataServiceImpl ===

internal::LocalDataServiceImpl::LocalDataServiceImpl(Library& lib) :
	library(lib -> addRef()),
	fileBackedMemory(kj::newDiskFilesystem()->getCurrent().clone()),
	dbCache(kj::refcounted<odb::DBCache>())
{}

Own<internal::LocalDataServiceImpl> internal::LocalDataServiceImpl::addRef() {
	return kj::addRef(*this);
}

void internal::LocalDataServiceImpl::setLimits(LocalDataService::Limits newLimits) {
	limits = newLimits;
}


void internal::LocalDataServiceImpl::setChunkDebugMode() {
	debugChunks = true;
}

Promise<Maybe<LocalDataRef<capnp::AnyPointer>>> internal::LocalDataServiceImpl::unwrap(DataRef<capnp::AnyPointer>::Client src) {
	return serverSet.getLocalServer(src).then([this, src](Maybe<DataRef<capnp::AnyPointer>::Server&> maybeServer) mutable -> Maybe<LocalDataRef<capnp::AnyPointer>> {
		KJ_IF_MAYBE(server, maybeServer) {
			#if KJ_NO_RTTI
				auto backend = static_cast<internal::LocalDataRefImpl*>(server);
			#else
				auto backend = dynamic_cast<internal::LocalDataRefImpl*>(server);
				KJ_REQUIRE(backend != nullptr);
			#endif

			// If yes, extract the backend and return it
			return LocalDataRef<capnp::AnyPointer>(backend -> addRef(), this -> serverSet);
		} else {
			return nullptr;
		}
	}).attach(addRef());
}

struct internal::LocalDataServiceImpl::DataRefDownloadProcess : public DownloadTask<LocalDataRef<capnp::AnyPointer>> {
	Own<LocalDataServiceImpl> service;
	bool recursive;
	
	Own<const LocalDataStore::Entry> dataEntry;
	
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
		
		return service -> download(ref.castAs<DataRef<capnp::AnyPointer>>(), true, this -> ctx);
	}
	
	virtual Promise<Maybe<ResultType>> useCached() override {
		auto dataHash = metadata.getDataHash();
		if(dataHash.size() > 0) {
			auto lStore = getActiveThread().library() -> store().lockShared();
			
			KJ_IF_MAYBE(rowPtr, lStore -> table.find(dataHash)) {
				dataEntry = (*rowPtr) -> addRef();
				return buildResult()
				.then([](ResultType result) {
					return Maybe<ResultType>(mv(result));
				});
			}
		}
		
		return Maybe<ResultType>(nullptr);
	}
	
	Promise<void> beginDownload() override {
		downloadBuffer = kj::heapArray<kj::byte>(metadata.getDataSize());
		downloadOffset = 0;
		return READY_NOW;
	}
	
	Promise<void> receiveData(kj::ArrayPtr<const kj::byte> data) override {
		KJ_REQUIRE(downloadOffset + data.size() <= downloadBuffer.size());
		memcpy(downloadBuffer.begin() + downloadOffset, data.begin(), data.size());
		
		downloadOffset += data.size();
		return READY_NOW;
	}
	
	Promise<void> finishDownload() override {
		KJ_REQUIRE(downloadOffset == downloadBuffer.size());
		KJ_REQUIRE(downloadBuffer != nullptr);
		
		dataEntry = kj::atomicRefcounted<const LocalDataStore::Entry>(
			metadata.getDataHash(),	mv(downloadBuffer)
		);
		
		auto lStore = getActiveThread().library() -> store().lockExclusive();
			
		KJ_IF_MAYBE(rowPtr, lStore -> table.find(metadata.getDataHash())) {
			// If found now, discard current download
			// Note: This should happen only rarely
			dataEntry = (*rowPtr) -> addRef();
		} else {
			// If not found, store row
			lStore -> table.insert(dataEntry -> addRef());
		}
		
		return READY_NOW;
	}
	
	Promise<ResultType> buildResult() override {
		auto backend = kj::refcounted<internal::LocalDataRefImpl>();
		
		// Build reader capability table
		auto capHooks = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(capTable.size());
		for(auto client : capTable) {
			Own<capnp::ClientHook> hookPtr = capnp::ClientHook::from(mv(client));
			
			if(hookPtr.get() != nullptr)
				capHooks.add(mv(hookPtr));
			else
				capHooks.add(nullptr);
		}
	
		backend->readerTable = kj::heap<capnp::ReaderCapabilityTable>(capHooks.finish());
		backend->capTableClients = mv(capTable);
		
		backend->_metadata.setRoot(metadata.asReader());
		
		backend->entryRef = mv(dataEntry);
		
		return ResultType(mv(backend), service -> serverSet);
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

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publish(DataRefMetadata::Reader metaData, Array<const byte>&& data, ArrayPtr<Maybe<Own<capnp::ClientHook>>> capTable) {
	// Check if we have the id already, if not, 
	Own<const LocalDataStore::Entry> entry;
	
	KJ_REQUIRE(data.size() >= metaData.getDataSize(), "Data do not fit inside provided array");
	KJ_REQUIRE(metaData.getCapTableSize() == capTable.size(), "Specified capability count must match provided table");
	
	auto backend = kj::refcounted<internal::LocalDataRefImpl>();
	
	backend -> _metadata.setRoot(metaData);
	auto myMetaData = backend -> _metadata.getRoot<DataRefMetadata>();
	
	// Check if hash is nullptr. If yes, construct own.
	if(metaData.getDataHash().size() == 0) {
		auto hashFunction = getActiveThread().library() -> defaultHash();
		hashFunction -> update(data.begin(), data.size());
		
		KJ_STACK_ARRAY(uint8_t, hashOutput, hashFunction -> output_length(), 1, 64);
		hashFunction -> final(hashOutput.begin());
		
		myMetaData.setDataHash(hashOutput);
	}
	
	// Prepare construction of the data
	{
		kj::Locked<LocalDataStore> lStore = library -> store().lockExclusive();
		auto dataHash = myMetaData.getDataHash();
		KJ_IF_MAYBE(ppRow, lStore -> table.find(dataHash)) {
			entry = (*ppRow) -> addRef();
		} else {
			entry = kj::atomicRefcounted<LocalDataStore::Entry>(dataHash, mv(data));
			lStore -> table.insert(entry -> addRef());
		}
	}
	
	// Prepare some clients
	auto clients   = kj::heapArrayBuilder<capnp::Capability::Client>    (capTable.size());
	auto tableCopy = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(capTable.size());
	
	for(size_t i = 0; i < capTable.size(); ++i) {
		KJ_IF_MAYBE(hook, capTable[i]) {
			clients.add((*hook) -> addRef());
			tableCopy.add((*hook) -> addRef());
		} else {
			clients.add(nullptr);
			tableCopy.add(nullptr); // note that this does not add a nullptr own inside the maybe, but an empty maybe
		}
	}
	
	// Prepare backend
	backend->readerTable = kj::heap<capnp::ReaderCapabilityTable>(tableCopy.finish());
	backend->capTableClients = clients.finish();
	backend->entryRef = mv(entry);
		
	// Now construct a local data ref from the backend
	return LocalDataRef<capnp::AnyPointer>(backend->addRef(), this -> serverSet);
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
				auto childHashReq = thisCap().hashRequest<capnp::AnyPointer>();
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
		Own<const kj::WritableFileMapping> mapping;
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
		bl -> mapping = file.mmapWritable((TOTAL_PREFIX_SIZE + dataSize) / WORDS * sizeof(word), nWords / WORDS * sizeof(word));
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
					return finalize(*pResult)
					.then([](uint64_t x) -> Maybe<uint64_t> {
						return x;
					});
				}
				
				auto dataHash = metadata.getDataHash().asConst();
				
				// Get a ref to the store entry if it is found
				Maybe<Own<const LocalDataStore::Entry>> maybeStoreEntry = nullptr;
				if(dataHash.size() > 0) {
					auto lStore = getActiveThread().library() -> store().lockShared();
					
					KJ_IF_MAYBE(rowPtr, lStore -> table.find(dataHash)) {
						maybeStoreEntry = (*rowPtr) -> addRef();
					}
				}
				
				KJ_IF_MAYBE(pStoreEntry, maybeStoreEntry) {
					// We have the block locally. Just allocate space for it and memcpy
					// it over
					const LocalDataStore::Entry& storeEntry = **pStoreEntry;
					
					auto data = storeEntry.value.asPtr();
					metadata.setDataSize(data.size());
					
					auto& dataRecord = parent.allocData(data.size());
					kj::byte* target = dataRecord.mapping -> get().begin();
					
					memcpy(target, data.begin(), data.size());
					dataRecord.mapping -> sync(dataRecord.mapping -> get());
					
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
				kj::ArrayPtr<kj::byte> output = pBlock -> mapping -> get();
				KJ_REQUIRE(writeOffset + data.size() <= pBlock -> sizeBytes, "Overflow in download");
				memcpy(output.begin() + writeOffset, data.begin(), data.size());
				writeOffset += data.size();
				
				return READY_NOW;
			} else {
				KJ_FAIL_REQUIRE("Failed to allocate data");
			}
		}
		
		Promise<void> finishDownload() override { return READY_NOW; }
		
		Promise<uint64_t> buildResult() override {
			KJ_IF_MAYBE(pBlock, block) {
				kj::ArrayPtr<kj::byte> output = pBlock -> mapping -> get();
				if(writeOffset < output.size()) {
					memset(output.begin() + writeOffset, 0, output.size() - writeOffset);
				}
				
				pBlock -> mapping -> sync(output);
				
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
	
	// Scan all refs, looking to feed out root and fulfill any dependencies
	for(auto i : kj::indices(objectInfo)) {
		auto object = objectInfo[i];
		auto refs = object.getRefs();
		
		auto refBuilder = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(refs.size());
		for(auto iRef : kj::indices(refs)) {
			auto refInfo = refs[iRef];
			if(refInfo.isNull()) {
				refBuilder.add(nullptr);
			} else if(refInfo.isException()) {
				refBuilder.add(capnp::ClientHook::from(capnp::Capability::Client(
					fromProto(refInfo.getException())
				)));
			} else if(refInfo.isObject()) {
				refBuilder.add(capnp::ClientHook::from(capnp::Capability::Client(
					refPromises[refInfo.getObject()].addBranch()
				)));
			} else {
				refBuilder.add(capnp::ClientHook::from(capnp::Capability::Client(
					KJ_EXCEPTION(DISCONNECTED, "Unknown object type")
				)));
			}
		}
		
		auto dataRecord = dataInfo[object.getDataId()];
		
		kj::Array<const kj::byte> dataMapping = f.mmap((dataOffset / WORDS + dataRecord.getOffsetWords()) * sizeof(word), dataRecord.getSizeBytes());
		
		LocalDataRef<capnp::AnyPointer> published = publish(
			object.getMetadata(),
			mv(dataMapping),
			refBuilder.finish()
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
	auto writer = heapHeld<ArchiveWriter>(out);
	
	return writer -> writeArchive(mv(ref)).attach(writer.x());
}


Promise<void> internal::LocalDataServiceImpl::clone(CloneContext context) {
	return dbCache -> cache(context.getParams().getSource())
	.then([context](DataRef<capnp::AnyPointer>::Client result) mutable {
		context.getResults().setRef(mv(result));
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
		data = kj::heapArray<byte>(inData.getAs<capnp::Data>());
	} else {
		// typeID != 0 indicates struct or capability
		
		// Check format
		// Only structs and capabilities can be typed
		// Lists have no corresponding schema nodes (and therefore no IDs)
		KJ_REQUIRE(inData.isStruct() || inData.isCapability());
		
		capnp::MallocMessageBuilder mb;
		AnyPointer::Builder root = capTable.imbue(mb.initRoot<AnyPointer>());
		root.set(inData);
		
		data = wordsToBytes(capnp::messageToFlatArray(mb));
	}
		
	Temporary<DataRefMetadata> metaData;
	metaData.setId(params.getId());
	
	if(schema.isNull())
		metaData.getFormat().setRaw();
	else
		metaData.getFormat().initSchema().set(schema);
	metaData.setCapTableSize(capTable.getTable().size());
	metaData.setDataSize(data.size());

	auto ref = publish(
		metaData,
		mv(data),
		capTable.getTable()
	);
	
	auto cachedRef = dbCache -> cache(mv(ref));
	context.getResults().setRef(mv(cachedRef));
	
	return READY_NOW;
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
		
		return { context.tailCall(mv(request)), false, false };
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
	auto data = heapHeld<FlatData>(*this);
	
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
	})
	.attach(data.x());
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
	
	// Fill in references
	for(auto i : kj::range(0, nRefs)) {
		auto refInfo = info.getObjects()[i];
		auto links = refInfo.getRefs();
		
		auto capTableBuilder = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(links.size());
		for(auto iLink : kj::indices(links)) {
			auto link = links[iLink];
			
			if(link.isNull()) {
				capTableBuilder.add(nullptr);
				continue;
			}
			
			DataRef<>::Client target(nullptr);
			if(link.isException()) {
				target = DataRef<>::Client(fromProto(link.getException()));
			} else if(link.isObject()) {
				KJ_REQUIRE(link.getObject() < refs.size(), "Out of bounds link");
				target = DataRef<>::Client(refs[link.getObject()].addBranch());
			} else {
				KJ_FAIL_REQUIRE("Unknown link type encountered in flat representation");
			}
			
			capTableBuilder.add(capnp::ClientHook::from(mv(target)));
		}
		
		auto dataId = refInfo.getDataId();
		
		auto& mp = maybePublished[dataId];
		kj::Array<const byte> refData(nullptr);
		
		if(mp.is<kj::Array<const byte>>()) {
			refData = mv(mp.get<kj::Array<const byte>>());
		} else {
			refData = mp.get<LocalDataRef<>>().forkRaw();
		}
		
		auto publishedRef = publish(refInfo.getMetadata(), mv(refData), capTableBuilder.finish());
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

} // namespace fsc
