#include <kj/array.h>
#include <capnp/generated-header-support.h>
#include <capnp/serialize.h>
#include <capnp/orphan.h>
#include <kj/filesystem.h>
#include <kj/map.h>

#include "data.h"

namespace {
}

namespace fsc {
	
// === functions in internal ===

template<>
capnp::Data::Reader internal::getDataRefAs<capnp::Data>(internal::LocalDataRefImpl& impl) {
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
	context.getResults().setData(get<capnp::Data>());
	
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

capnp::FlatArrayMessageReader& internal::LocalDataRefImpl::ensureReader() {
	KJ_IF_MAYBE(reader, maybeReader) {
		return *reader;
	}
	
	// Obtain data as a byte pointer (note that this drops all attached objects to keep alive0
	ArrayPtr<const byte> bytePtr = getDataRefAs<capnp::Data>(*this);
	
	// Cast the data to a word array (let's hope they are aligned properly)
	ArrayPtr<const capnp::word> wordPtr = ArrayPtr<const capnp::word>(
		reinterpret_cast<const capnp::word*>(bytePtr.begin()),
		bytePtr.size() / sizeof(capnp::word)
	);
	
	return maybeReader.emplace(wordPtr);
}

// === class internal::LocalDataServiceImpl ===

internal::LocalDataServiceImpl::LocalDataServiceImpl(Library& lib) :
	library(lib -> addRef()),
	downloadPool(65536)
{}

Own<internal::LocalDataServiceImpl> internal::LocalDataServiceImpl::addRef() {
	return kj::addRef(*this);
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

Promise<LocalDataRef<capnp::AnyPointer>> internal::LocalDataServiceImpl::doDownload(DataRef<capnp::AnyPointer>::Client src, bool recursive) {
	// Use a fiber for downloading
	
	return downloadPool.startFiber([this, recursive, src](kj::WaitScope& ws) mutable {
		// Retrieve metadata and cap table
		auto metadataPromise = src.metadataRequest().send();
		auto capTablePromise = src.capTableRequest().send();
		
		// Process the capability table
		// We do this first, because the cap table might contain links to other datarefs
		// which we have to download as well
		auto capTableResponse = capTablePromise.wait(ws); // This needs to be kept alive
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
		
		// Now wait for the metadata to arrive.
		// We use this to check whether we need to download any data at all (as we might)
		// still have a copy of the data alive
		auto metadataResponse = metadataPromise.wait(ws); // This needs to be kept alive
		auto metadata = metadataResponse.getMetadata();
		
		Own<const LocalDataStore::Entry> entry;
		
		// Check if we have the ID already
		{
			auto lStore = library -> store.lockShared();
			KJ_IF_MAYBE(rowPtr, lStore -> table.find(metadata.getId())) {
				entry = (*rowPtr) -> addRef();
			}
		}
		
		// If id not found, we need to download
		if(entry.get() == nullptr) {
			// Block this task (not the thread itself, we are in a fiber) until
			// data have arrived.
			auto rawBytes = src.rawBytesRequest().send().wait(ws);
			
			// Copy the data into a heap buffer
			entry = kj::atomicRefcounted<LocalDataStore::Entry>(
				metadata.getId(),
				kj::heapArray<const byte>(rawBytes.getData())
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
		}
		
		// Initialize the backend struct with everything
		auto backend = kj::refcounted<internal::LocalDataRefImpl>();
		backend->readerTable = kj::heap<capnp::ReaderCapabilityTable>(mv(capHooks));
		backend->capTableClients = capClients.finish();
		backend->entryRef = mv(entry);
		
		backend->_metadata.setRoot(metadata);
				
		// Now construct a local data ref from the backend
		return LocalDataRef<capnp::AnyPointer>(backend->addRef(), this -> serverSet);
	});
}

Promise<void> internal::LocalDataServiceImpl::buildArchive(DataRef<capnp::AnyPointer>::Client root, Archive::Builder out) {
	using RemoteRef = DataRef<capnp::AnyPointer>::Client;
	using LocalRef  = LocalDataRef<capnp::AnyPointer>;
	
	using kj::TaskSet;
	using capnp::Orphan;
	
	using AnyArchive = Archive;
	
	return downloadPool.startFiber([this, root, out](kj::WaitScope& ws) mutable {		
		auto orphanage = capnp::Orphanage::getForMessageContaining(out);
		kj::TreeMap<ID, Orphan<AnyArchive::Entry>> entries;
				
		std::function<Promise<AnyArchive::Entry::Builder>(RemoteRef)> createEntry;
		createEntry = [&, this](RemoteRef ref) -> Promise<AnyArchive::Entry::Builder> {
			return download(ref, false)
			.then([&](LocalRef local) -> Promise<AnyArchive::Entry::Builder> {
				auto id = local.getID();
				
				// The ref is already stored locally. Return the entry
				KJ_IF_MAYBE(entry, entries.find(id)) {
					return entry -> get();
				}
				
				auto capTable = local.getCapTable();
				
				// The ref is not yet stored. We need to create a new entry.
				auto newOrphan = orphanage.newOrphan<AnyArchive::Entry>();
				auto newEntry = newOrphan.get();
				
				// Store entry for orphan in entry table
				entries.insert(id, mv(newOrphan));
				
				newEntry.setId(id);
				newEntry.setData(local.getRaw());
				newEntry.setTypeId(local.getTypeID());
				newEntry.initCapabilities((unsigned int) capTable.size());
				
				auto subDownloads = kj::heapArrayBuilder<Promise<void>>(capTable.size());
				
				// Try to inspect child references
				for(unsigned int i = 0; i < capTable.size(); ++i) {
					auto capRef = newEntry.getCapabilities()[i];
					
					subDownloads.add(
						// Try to download sub references and reference them in table
						createEntry(capTable[i].castAs<DataRef<capnp::AnyPointer>>())
						.then([capRef](AnyArchive::Entry::Builder b) mutable -> void {
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
		};
		
		AnyArchive::Entry::Builder rootEntry = createEntry(root).wait(ws);
				
		unsigned int offset = 0;
		out.initExtra((unsigned int) (entries.size() - 1));
		
		for(auto& mapEntry : entries) {
			Orphan<AnyArchive::Entry>& orphan = mapEntry.value;
			
			if(orphan.get().getId() == rootEntry.getId())
				out.adoptRoot(mv(orphan));
			else
				out.getExtra().adoptWithCaveats(offset++, mv(orphan));
		}
	}).attach(this -> addRef());
}

Promise<void> internal::LocalDataServiceImpl::writeArchive(DataRef<capnp::AnyPointer>::Client ref, kj::File& out) {
	Own<capnp::MallocMessageBuilder> msg;
	auto archive = msg -> initRoot<Archive>();
	
	return buildArchive(ref, archive)
	.then([&out, msg = mv(msg)]() mutable {
		// If the file has a file descriptor, we can directly write to it
		KJ_IF_MAYBE(fd, out.getFd()) {
			capnp::writeMessageToFd(*fd, *msg);
			return;
		}
		
		// If not, we will have to allocate a local copy and memcpy it over. Blergh.
		kj::Array<const byte> data = wordsToBytes(capnp::messageToFlatArray(*msg));
		out.writeAll(data);
	});
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
		
		LocalRef local = publish(
			entry.getId(),
			kj::heapArray<const byte>(entry.getData()),
			capTableBuilder.finish(),
			entry.getTypeId()
		);
		
		KJ_IF_MAYBE(target, fulfillers.find(entry.getId())) {
			(*target) -> fulfill(cp(local));
		} else {
			KJ_FAIL_ASSERT();
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

} // namespace fsc
