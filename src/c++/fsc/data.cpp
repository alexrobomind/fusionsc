#include <kj/array.h>
#include <capnp/generated-header-support.h>
#include <capnp/serialize.h>
#include <capnp/orphan.h>
#include <kj/filesystem.h>
#include <kj/map.h>

#include <functional>

#include "data.h"

namespace fsc {
	
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

void LocalDataService::setLimits(Limits newLimits) { impl->setLimits(newLimits); }

void LocalDataService::setChunkDebugMode() {
	impl -> setChunkDebugMode();
}

/*LocalDataRef<capnp::Data> LocalDataService::publish(ArrayPtr<const byte> id, Array<const byte>&& data) {
	return impl->publish(
		id,
		mv(data),
		kj::heapArray<Maybe<Own<capnp::ClientHook>>>(0),
		internal::capnpTypeId<capnp::Data>()
	).as<capnp::Data>();
}*/

// === class internal::LocalDataRefImpl ===

Own<internal::LocalDataRefImpl> internal::LocalDataRefImpl::addRef() {
	return kj::addRef(*this);
}

DataRef<capnp::AnyPointer>::Metadata::Reader internal::LocalDataRefImpl::localMetadata() {
	return _metadata.getRoot<Metadata>();
}

Promise<void> internal::LocalDataRefImpl::metadata(MetadataContext context) {
	context.getResults().setMetadata(localMetadata());
	
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

Promise<void> internal::LocalDataRefImpl::capTable(CapTableContext context) {
	using CC = capnp::Capability::Client;
	
	auto results = context.getResults();
	results.initTable((unsigned int) capTableClients.size());
	
	for(unsigned int i = 0; i < capTableClients.size(); ++i) {
		results.getTable().set(i, capTableClients[i]);
	}
	
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
			
			auto slice = data.slice(start, end);
			
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

// === class internal::LocalDataServiceImpl ===

internal::LocalDataServiceImpl::LocalDataServiceImpl(Library& lib) :
	library(lib -> addRef()),
	downloadPool(65536),
	fileBackedMemory(kj::newDiskFilesystem()->getCurrent().clone())
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

Promise<LocalDataRef<capnp::AnyPointer>> internal::LocalDataServiceImpl::download(DataRef<capnp::AnyPointer>::Client src, bool recursive) {
	// Check if the capability is actually local
	return serverSet.getLocalServer(src).then([src, recursive, this](Maybe<DataRef<capnp::AnyPointer>::Server&> maybeServer) mutable -> Promise<LocalDataRef<capnp::AnyPointer>> {
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
			// If not, download for real
			return doDownload(src, recursive);
		}
	});
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publish(ArrayPtr<const byte> id, Array<const byte>&& data, ArrayPtr<Maybe<Own<capnp::ClientHook>>> capTable, uint64_t cpTypeId) {
	// Check if we have the id already, if not, 
	Own<const LocalDataStore::Entry> entry;
		
	// Prepare construction of the data
	{
		kj::Locked<LocalDataStore> lStore = library -> store.lockExclusive();
		KJ_IF_MAYBE(ppRow, lStore -> table.find(id)) {
			entry = (*ppRow) -> addRef();
		} else {
			entry = kj::atomicRefcounted<LocalDataStore::Entry>(id, mv(data));
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
	
	// Prepare metadata
	auto backend = kj::refcounted<internal::LocalDataRefImpl>();
	backend->readerTable = kj::heap<capnp::ReaderCapabilityTable>(tableCopy.finish());
	backend->capTableClients = clients.finish();
	backend->entryRef = mv(entry);
	
	auto metadata = backend->_metadata.initRoot<DataRef<capnp::AnyPointer>::Metadata>();
	metadata.setId(id);
	metadata.setTypeId(cpTypeId);
	metadata.setCapTableSize(capTable.size());
	metadata.setDataSize(backend -> entryRef -> value.size());
		
	// Now construct a local data ref from the backend
	return LocalDataRef<capnp::AnyPointer>(backend->addRef(), this -> serverSet);
}

namespace {
	struct TransmissionReceiver : public DataRef<capnp::AnyPointer>::Receiver::Server {
		kj::ArrayPtr<kj::byte> target;
		size_t offset;
		
		TransmissionReceiver(kj::ArrayPtr<kj::byte> target) :
			target(target), offset(0)
		{}
		
		Promise<void> begin(BeginContext context) override {
			KJ_REQUIRE(offset == 0);
			KJ_REQUIRE(context.getParams().getNumBytes() == target.size());
			
			return READY_NOW;
		}
		
		Promise<void> receive(ReceiveContext context) override {
			auto data = context.getParams().getData();
			
			KJ_REQUIRE(offset + data.size() <= target.size());
			memcpy(target.begin() + offset, data.begin(), data.size());
			
			offset += data.size();
			
			return READY_NOW;
		}
		
		Promise<void> done(DoneContext context) override {
			KJ_REQUIRE(offset == target.size());
			
			return READY_NOW;
		}
	};
}

Promise<LocalDataRef<capnp::AnyPointer>> internal::LocalDataServiceImpl::doDownload(DataRef<capnp::AnyPointer>::Client src, bool recursive) {
	using RemoteRef = DataRef<capnp::AnyPointer>;
	using capnp::Response;
	using EntryPromise = Promise<Own<const LocalDataStore::Entry>>;
		
	// Allocate backend struct
	auto backend = kj::refcounted<internal::LocalDataRefImpl>();
	
	// Sub-process 1: Download capabilities
	auto downloadCaps = src.capTableRequest().send()
	.then([backend = backend->addRef(), recursive, this](Response<RemoteRef::CapTableResults> capTableResponse) mutable {
		auto capTable = capTableResponse.getTable();
		
		auto capHooks = kj::heapArray<Maybe<Own<capnp::ClientHook>>>(capTable.size());
		for(unsigned int i = 0; i < capTable.size(); ++i) {
			capnp::Capability::Client client = capTable[i];
			
			// Check if we need to initiate any child download tasks
			if(recursive) {		
				// Download the capability table entry
				// If the entry does not support the DataRef interface,
				// then just return it without complaining
				// All other errors get passed through
				client = ((Promise<capnp::Capability::Client>) doDownload(
					client.castAs<DataRef<capnp::AnyPointer>>(),
					true
				)).catch_([client](kj::Exception&& e) mutable {				
					if(e.getType() == kj::Exception::Type::UNIMPLEMENTED)
						return client;
					throw e;
				});
			}
			
			Own<capnp::ClientHook> hookPtr = capnp::ClientHook::from(client);
			
			if(hookPtr.get() != nullptr)
				capHooks[i] = mv(hookPtr);
		}
		
		// Store a copy of the list
		auto capClients = kj::heapArrayBuilder<capnp::Capability::Client>(capTable.size());
		for(unsigned int i = 0; i < capTable.size(); ++i) {
			capClients.add(capTable[i]);
		}
		
		backend->readerTable = kj::heap<capnp::ReaderCapabilityTable>(mv(capHooks));
		backend->capTableClients = capClients.finish();
	});
	
	// Sub-process 2: Download metadata and (if neccessary) data
	auto downloadData = src.metadataRequest().send()
	.then([backend = backend->addRef(), src, this](Response<RemoteRef::MetadataResults> metadataResponse) mutable -> EntryPromise {
		auto metadata = metadataResponse.getMetadata();
		backend->_metadata.setRoot(metadata);
				
		// Check if we have the ID already
		{
			auto lStore = library -> store.lockShared();
			KJ_IF_MAYBE(rowPtr, lStore -> table.find(metadata.getId())) {
				return (*rowPtr) -> addRef();
			}
		}
		
		kj::Vector<Promise<void>> processPromises;
		auto dataSize = metadata.getDataSize();
		auto buffer = kj::heapArray<kj::byte>(dataSize);
		
		/*constexpr size_t CHUNK_SIZE = 1024 * 1024;
		for(size_t start = 0; start < dataSize; start += CHUNK_SIZE) {
			size_t end = start + CHUNK_SIZE;
			if(end > dataSize)
				end = dataSize;
			
			auto request = src.rawBytesRequest();
			request.setStart(start);
			request.setEnd(end);
			
			auto processPromise = request.send().then([out = buffer.slice(start, end)](Response<RemoteRef::RawBytesResults> rawBytesResponse) mutable {
				auto data = rawBytesResponse.getData();
				
				KJ_REQUIRE(data.size() == out.size());
				memcpy(out.begin(), data.begin(), out.size());				
			});
			
			processPromises.add(mv(processPromise));
		}
		
		return kj::joinPromises(processPromises.releaseAsArray())*/
		DataRef<capnp::AnyPointer>::Receiver::Client receiver = kj::heap<TransmissionReceiver>(buffer);
		
		auto transmitRequest = src.transmitRequest();
		transmitRequest.setStart(0);
		transmitRequest.setEnd(dataSize);
		transmitRequest.setReceiver(receiver);
		
		return transmitRequest.send().ignoreResult()		
		.then([backend = mv(backend), buffer = mv(buffer), this]() mutable {
			// Copy the data into a heap buffer
			auto metadata = backend->_metadata.getRoot<RemoteRef::Metadata>();
			
			auto entry = kj::atomicRefcounted<const LocalDataStore::Entry>(
				metadata.getId(),
				// kj::heapArray<const byte>(rawBytesResponse.getData())
				mv(buffer)
			);
			
			// Lock the store again, check for concurrent download and remember row
			{
				auto lStore = library -> store.lockExclusive();
				
				KJ_IF_MAYBE(rowPtr, lStore -> table.find(metadata.getId())) {
					// If found now, discard current download
					// Note: This should happen only rarely
					entry = (*rowPtr) -> addRef();
				} else {
					// If not found, store row
					lStore -> table.insert(entry -> addRef());
				}
			}
			
			return entry;
		});
	})
	.then([backend = backend->addRef()](Own<const LocalDataStore::Entry> entry) mutable {
		backend->entryRef = mv(entry);
	});
	
	auto allDoneBuilder = kj::heapArrayBuilder<Promise<void>>(2);
	allDoneBuilder.add(mv(downloadCaps));
	allDoneBuilder.add(mv(downloadData));
	
	auto allDone = kj::joinPromises(allDoneBuilder.finish());
	
	return allDone.then([this, backend = backend->addRef()]() mutable {
		return LocalDataRef<capnp::AnyPointer>(backend->addRef(), this -> serverSet);
	}).attach(this -> addRef(), backend -> addRef());
}

Promise<Archive::Entry::Builder> internal::LocalDataServiceImpl::createArchiveEntry(DataRef<capnp::AnyPointer>::Client ref, kj::TreeMap<ID, capnp::Orphan<Archive::Entry>>& entries, capnp::Orphanage orphanage, Maybe<Nursery&> nursery) {
	using RemoteRef = DataRef<capnp::AnyPointer>::Client;
	using LocalRef  = LocalDataRef<capnp::AnyPointer>;
	using capnp::Orphan;
	
	return download(ref, false).then([this, &entries, orphanage, nursery = Maybe<Nursery&>(nursery)](LocalDataRef<capnp::AnyPointer> local) mutable -> Promise<Archive::Entry::Builder> {
		auto id = local.getID();
		
		// The ref is already stored locally. Return the entry
		KJ_IF_MAYBE(entry, entries.find(id)) {
			return entry -> get();
		}
		
		auto capTable = local.getCapTable();
		
		// The ref is not yet stored. We need to create a new entry.
		auto newOrphan = orphanage.newOrphan<Archive::Entry>();
		auto newEntry = newOrphan.get();
		
		// Store entry for orphan in entry table
		entries.insert(id, mv(newOrphan));
		
		newEntry.setId(id);
		
		auto raw = local.getRaw();
		auto rawSize = raw.size();
		
		const size_t CHUNK_SIZE = debugChunks ? 64 : 1024 * 1024 * 256;
		const size_t nChunks = rawSize < CHUNK_SIZE ? 1 : rawSize / CHUNK_SIZE + 1;
		
		auto chunkData = kj::heapArray<capnp::Orphan<capnp::Data>>(nChunks);
		
		KJ_IF_MAYBE(pNursery, nursery) {
			pNursery -> add(kj::heap(local));
		}
		
		if(nChunks == 1) {
			// Don't add extra segments for data that fit inside a single chunk
			chunkData[0] = orphanage.newOrphanCopy<capnp::Data::Reader>(raw);
		} else {
			for(size_t iChunk = 0; iChunk < nChunks; ++iChunk) {
				size_t start = CHUNK_SIZE * iChunk;
				size_t end   = CHUNK_SIZE * (iChunk + 1);
				
				if(start >= rawSize)
					continue;
				
				if(end >= rawSize)
					end = rawSize;
				
				auto inputData = raw.slice(start, end);
				
				KJ_IF_MAYBE(pNursery, nursery) {
					// We need to make an intermediate copy of the data if they are not word-aligned
					// in length (not going to happen with capnp messages) to add the missing bytes.
					
					if(inputData.size() % sizeof(capnp::word) != 0) {
						size_t bufSize = inputData.size() / 8;
						bufSize += 1;
						bufSize *= 8;
						
						auto heapBuffer = kj::heapArray<kj::byte>(bufSize);
						
						memset(heapBuffer.begin(), 0, heapBuffer.size());
						memcpy(heapBuffer.begin(), inputData.begin(), inputData.size());
						
						inputData = kj::ArrayPtr<const kj::byte>(heapBuffer.begin(), inputData.size());
						
						pNursery -> add(kj::heap(mv(heapBuffer)));
					}
					
					chunkData[iChunk] = orphanage.referenceExternalData(inputData);
				} else {
					chunkData[iChunk] = orphanage.newOrphanCopy<capnp::Data::Reader>(inputData);
				}
			}
		}
		
		auto outData = newEntry.initData(nChunks);
		for(size_t iChunk = 0; iChunk < nChunks; ++iChunk)
			outData.adopt(iChunk, mv(chunkData[iChunk]));
		
		newEntry.setTypeId(local.getTypeID());
		newEntry.initCapabilities((unsigned int) capTable.size());
		
		auto subDownloads = kj::heapArrayBuilder<Promise<void>>(capTable.size());
		
		// Try to inspect child references
		for(unsigned int i = 0; i < capTable.size(); ++i) {
			auto capRef = newEntry.getCapabilities()[i];
			
			subDownloads.add(
				// Try to download sub references and reference them in table
				createArchiveEntry(capTable[i].castAs<DataRef<capnp::AnyPointer>>(), entries, orphanage, nursery)
				.then([capRef](Archive::Entry::Builder b) mutable -> void {
					capRef.getDataRefInfo().setRefID(b.getId());
				})
				
				// If download fails due to UNIMPLEMENTED, remember that this 
				// is no data ref.
				.catch_([capRef](kj::Exception&& e) mutable -> void {
					if(e.getType() != kj::Exception::Type::UNIMPLEMENTED)
						throw e;
					
					capRef.getDataRefInfo().setNoDataRef();
				})
			);
		}
		
		return kj::joinPromises(subDownloads.finish()).then([newEntry]() { return newEntry; });
	});
}

Promise<void> internal::LocalDataServiceImpl::buildArchive(DataRef<capnp::AnyPointer>::Client root, Archive::Builder out, Maybe<Nursery&> nursery) {
	using kj::TaskSet;
	
	using AnyArchive = Archive;
	
	auto orphanage = capnp::Orphanage::getForMessageContaining(out);
	auto entries = kj::heap<kj::TreeMap<ID, capnp::Orphan<AnyArchive::Entry>>>();
	
	auto task = createArchiveEntry(root, *entries, orphanage, nursery)
	.then([out, &entries = *entries](Archive::Entry::Builder rootEntry) mutable {
		unsigned int offset = 0;
		out.initExtra((unsigned int) (entries.size() - 1));
		
		for(auto& mapEntry : entries) {
			capnp::Orphan<Archive::Entry>& orphan = mapEntry.value;
			
			if(orphan.get().getId() == rootEntry.getId())
				out.adoptRoot(mv(orphan));
			else
				out.getExtra().adoptWithCaveats(offset++, mv(orphan));
		}
	});
	
	return task.attach(this -> addRef(), mv(entries));
}

Promise<void> internal::LocalDataServiceImpl::writeArchive(DataRef<capnp::AnyPointer>::Client ref, const kj::File& out) {
	auto msg = kj::heap<capnp::MallocMessageBuilder>();
	auto archive = msg -> initRoot<Archive>();
	
	auto nursery = kj::heap<Nursery>();
	
	return buildArchive(ref, archive, *nursery)
	.then([&out, msg = mv(msg)]() mutable {
		// If the file has a file descriptor, we can directly write to it
		KJ_IF_MAYBE(fd, out.getFd()) {
			capnp::writeMessageToFd(*fd, *msg);
			return;
		}
		
		// If not, we will have to allocate a local copy and memcpy it over. Blergh.
		kj::Array<const byte> data = wordsToBytes(capnp::messageToFlatArray(*msg));
		out.writeAll(data);
	}).attach(mv(nursery));
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publishArchive(Archive::Reader archive) {
	using RemoteRef = DataRef<capnp::AnyPointer>::Client;
	using LocalRef  = LocalDataRef<capnp::AnyPointer>;
	
	kj::TreeMap<ID, Own<kj::PromiseFulfiller<LocalRef>>> fulfillers;
	kj::TreeMap<ID, RemoteRef> placeholderClients;
	
	kj::TreeMap<ID, Archive::Entry::Reader> entries;
	
	auto prepareEntry = [&](Archive::Entry::Reader entry) {
		// Create promise-based clients for remote reference resolution
		auto pfPair = kj::newPromiseAndFulfiller<LocalRef>();
		fulfillers.insert(entry.getId(), mv(pfPair.fulfiller));
		placeholderClients.insert(entry.getId(), mv(pfPair.promise));
		
		entries.insert(entry.getId(), entry);
	};
	
	auto handleEntry = [&, this](Archive::Entry::Reader entry) {
		auto capInfoTable = entry.getCapabilities();
		auto capTableBuilder = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(capInfoTable.size());
		
		for(auto capInfo : capInfoTable) {
			if(capInfo.getDataRefInfo().isRefID()) {
				KJ_IF_MAYBE(client, placeholderClients.find(capInfo.getDataRefInfo().getRefID())) {
					capTableBuilder.add(capnp::ClientHook::from(*client));
				} else {
					KJ_FAIL_REQUIRE("Referenced data ref missing in archive");
				}
			} else {
				capTableBuilder.add(nullptr);
			}
		}
		
		kj::Array<kj::byte> data;
		auto inData = entry.getData();
		
		size_t totalSize = 0;
		for(auto chunk : inData)
			totalSize += chunk.size();
		
		data = kj::heapArray<kj::byte>(totalSize);
		
		size_t start = 0;
		for(auto chunk : inData) {
			size_t end = start + chunk.size();
			auto out = data.slice(start, end);
			memcpy(out.begin(), chunk.begin(), chunk.size());
			start = end;
		}
		
		LocalRef local = publish(
			entry.getId(),
			mv(data),
			capTableBuilder.finish(),
			entry.getTypeId()
		);
		
		KJ_IF_MAYBE(target, fulfillers.find(entry.getId())) {
			(*target) -> fulfill(cp(local));
		} else {
			KJ_FAIL_ASSERT("Internally required fulfiller not found");
		}
		
		return local;
	};
	
	prepareEntry(archive.getRoot());
	for(auto e : archive.getExtra())
		prepareEntry(e);
	
	for(auto e : archive.getExtra())
		handleEntry(e);
	
	return handleEntry(archive.getRoot());
}

namespace {
	struct SharedArrayHolder : public kj::AtomicRefcounted {
		kj::Array<const capnp::word> data;
	};
}

LocalDataRef<capnp::AnyPointer> internal::LocalDataServiceImpl::publishArchive(const kj::ReadableFile & f, const capnp::ReaderOptions options) {
	using RemoteRef = DataRef<capnp::AnyPointer>::Client;
	using LocalRef  = LocalDataRef<capnp::AnyPointer>;
	
	auto arrayHolder = kj::atomicRefcounted<SharedArrayHolder>();
	
	// Open file globally to read the overview data
	auto fileData = f.stat();
	{
		kj::Array<const capnp::word> data = bytesToWords(f.mmap(0, fileData.size));
		arrayHolder->data = mv(data);
	}
	
	capnp::FlatArrayMessageReader reader(arrayHolder->data, options);
	auto archive = reader.getRoot<Archive>();
	
	kj::TreeMap<ID, Own<kj::PromiseFulfiller<LocalRef>>> fulfillers;
	kj::TreeMap<ID, RemoteRef> placeholderClients;
	
	kj::TreeMap<ID, Archive::Entry::Reader> entries;
	
	auto prepareEntry = [&](Archive::Entry::Reader entry) {
		// Create promise-based clients for remote reference resolution
		auto pfPair = kj::newPromiseAndFulfiller<LocalRef>();
		fulfillers.insert(entry.getId(), mv(pfPair.fulfiller));
		placeholderClients.insert(entry.getId(), mv(pfPair.promise));
		
		entries.insert(entry.getId(), entry);
	};
	
	auto handleEntry = [&, this](Archive::Entry::Reader entry) {
		auto capInfoTable = entry.getCapabilities();
		auto capTableBuilder = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(capInfoTable.size());
		
		for(auto capInfo : capInfoTable) {
			if(capInfo.getDataRefInfo().isRefID()) {
				KJ_IF_MAYBE(client, placeholderClients.find(capInfo.getDataRefInfo().getRefID())) {
					capTableBuilder.add(capnp::ClientHook::from(*client));
				} else {
					KJ_FAIL_REQUIRE("Referenced data ref missing in archive");
				}
			} else {
				capTableBuilder.add(nullptr);
			}
		}
		
		kj::Array<const kj::byte> outData;
		
		auto inData = entry.getData();
		
		// Data can be mmapped if they are only one chunk
		// or if the chunks are aligned consecutively in
		// the file.
		bool canMMap = true;
		for(size_t iChunk = 1; iChunk < inData.size(); ++iChunk) {
			auto thisChunk = inData[iChunk];
			auto prevChunk = inData[iChunk - 1];
			
			if(thisChunk.begin() != prevChunk.end())
				canMMap = false;
		}
		
		size_t totalSize = 0;
		for(auto chunk : inData)
			totalSize += chunk.size();
		
		if(debugChunks) {
			KJ_REQUIRE(canMMap);
		}
		
		//if(inData.size() == 1) {
		if(canMMap) {
			// Single-chunk data can be loaded by mmapping the file (allows on-demand loading)
			// auto onlyChunk = inData[0];
			// KJ_LOG(WARNING, onlyChunk.begin() - data.asBytes().begin(), onlyChunk.size(), f.stat().size);
			// outData = f.mmap(onlyChunk.begin() - data.asBytes().begin(), onlyChunk.size());
			// KJ_LOG(WARNING, "Mmap OK");			
			if(totalSize == 0)
				outData = kj::heapArray<const kj::byte>(0);
			else
				outData = kj::ArrayPtr<const kj::byte>(inData[0].begin(), totalSize).attach(kj::atomicAddRef(*arrayHolder));
			
		} else {
			KJ_LOG(WARNING, "Archive has non-memory-mappable chunk structure");
			
			// Non-aligned chunks have to be copied into memory manually
			auto mutableData = kj::heapArray<kj::byte>(totalSize);
			
			size_t start = 0;
			for(auto chunk : inData) {
				size_t end = start + chunk.size();
				auto out = mutableData.slice(start, end);
				memcpy(out.begin(), chunk.begin(), chunk.size());
				end = start;
			}
			outData = mv(mutableData);
		}
		
		// KJ_LOG(WARNING, "Publishing");
		LocalRef local = publish(
			entry.getId(),
			mv(outData),
			capTableBuilder.finish(),
			entry.getTypeId()
		);
		
		KJ_IF_MAYBE(target, fulfillers.find(entry.getId())) {
			(*target) -> fulfill(cp(local));
		} else {
			KJ_FAIL_ASSERT("Internally required fulfiller not found");
		}
		
		return local;
	};
	
	prepareEntry(archive.getRoot());
	for(auto e : archive.getExtra())
		prepareEntry(e);
	
	for(auto e : archive.getExtra())
		handleEntry(e);
	
	return handleEntry(archive.getRoot());
}

Promise<void> internal::LocalDataServiceImpl::clone(CloneContext context) {
	return download(context.getParams().getSource(), false)
	.then([context](LocalDataRef<capnp::AnyPointer> ref) mutable {
		context.getResults().setRef(ref);
	});
}

Promise<void> internal::LocalDataServiceImpl::store(StoreContext context) {
	using capnp::AnyPointer;
	using capnp::AnyList;
	using capnp::ElementSize;
	
	auto params = context.getParams();
	
	AnyPointer::Reader inData = params.getData();
	uint64_t typeId = params.getTypeId();
	
	Array<byte> data = nullptr;
	
	capnp::BuilderCapabilityTable capTable;
	
	if(typeId == 0) {
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
	
	auto ref = publish(
		params.getId(),
		mv(data),
		capTable.getTable(),
		typeId
	);
	
	context.getResults().setRef(ref);
	
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
		auto request = backend.typelessRequest(interfaceId, methodId, context.getParams().targetSize());
		request.set(context.getParams());
		context.releaseParams();
		
		return { context.tailCall(mv(request)), false };
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
	
	return typedClient.metadataRequest().send()
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

} // namespace fsc
