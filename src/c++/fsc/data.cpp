#include "data.h"

namespace fsc {

DataRef<capnp::AnyPointer>::Metadata::Reader internal::LocalDataRefImpl::localMetadata() {
	return _metadata.getRoot<Metadata>();
}

Promise<void> internal::LocalDataRefImpl::metadata(MetadataContext context) {
	context.getResults().setMetadata(localMetadata());
}

Promise<void> internal::LocalDataRefImpl::rawBytes(RawBytesContext context) {
	context.getResults().setData(*get<capnp::Data>());
}

Promise<void> internal::LocalDataRefImpl::capTable(CapTableContext context) {
	using CC = capnp::Capability::Client;
	
	auto results = context.getResults();
	results.initTable((unsigned int) capTableClients.size());
	
	for(unsigned int i = 0; i < capTableClients.size(); ++i) {
		results.getTable().set(i, capTableClients[i]);
	}
}

template<>
Own<capnp::Data::Reader> internal::LocalDataRefImpl::get<capnp::Data>() {
	return kj::heap<capnp::Data::Reader>(entryRef -> value.asPtr()).attach(addRef());
}

// === class LocalDataService::Impl ===

Promise<LocalDataRef<capnp::AnyPointer>> LocalDataService::Impl::download(DataRef<capnp::AnyPointer>::Client src) {
	// Check if the capability is actually local
	return serverSet.getLocalServer(src).then([src, this](Maybe<DataRef<capnp::AnyPointer>::Server> maybeServer) mutable -> Promise<LocalDataRef<capnp::AnyPointer>> {
		KJ_IF_MAYBE(server, maybeServer) {
			// If yes, extract the backend and return it
			return LocalDataRef<capnp::AnyPointer>(*static_cast<internal::LocalDataRefImpl*>(server), this -> serverSet);
		} else {
			// If not, download for real
			return doDownload(src);
		}
	});
}

LocalDataRef<capnp::AnyPointer> LocalDataService::Impl::publish(Array<byte> id, Array<byte>&& data, capnp::BuilderCapabilityTable&& capTable, uint64_t cpTypeId) {
	// Check if we have the id already, if not, 
	Own<const LocalDataStore::Entry> entry;
		
	// Prepare construction of the data
	{
		kj::Locked<LocalDataStore> lStore = backingStore.lockExclusive();
		KJ_IF_MAYBE(ppRow, lStore -> table.find(id)) {
			entry = (*ppRow) -> addRef();
		} else {
			entry = kj::atomicRefcounted<LocalDataStore::Entry>(id, mv(data));
		}
	}
	
	// Prepare some clients
	auto rawTable = kj::heapArray(capTable.getTable());
	auto clients = kj::heapArrayBuilder<capnp::Capability::Client>(rawTable.size());
	
	for(size_t i = 0; i < rawTable.size(); ++i) {
		KJ_IF_MAYBE(hook, rawTable[i]) {
			clients.add((*hook) -> addRef());
		} else {
			clients.add(nullptr);
		}
	}
	
	// Prepare metadata
	internal::LocalDataRefImpl backend;
	backend.readerTable = kj::heap<capnp::ReaderCapabilityTable>(rawTable);
	backend.capTableClients = clients.finish();
	backend.entryRef = mv(entry);
	
	auto metadata = backend._metadata.initRoot<DataRef<capnp::AnyPointer>::Metadata>();
	metadata.setId(id);
	metadata.setTypeId(cpTypeId);
	metadata.setCapTableSize(rawTable.size());
	metadata.setDataSize(entry -> value.size());
		
	// And move it into a refcounted heap instance
	Own<internal::LocalDataRefImpl> backendRef = kj::refcounted<internal::LocalDataRefImpl>(mv(backend));
	
	// Now construct a local data ref from the backend
	return LocalDataRef<capnp::AnyPointer>(*backendRef, this -> serverSet);
}

Promise<LocalDataRef<capnp::AnyPointer>> LocalDataService::Impl::doDownload(DataRef<capnp::AnyPointer>::Client src) {
	// Use a fiber for downloading
	
	return kj::startFiber(65536, [this, src](kj::WaitScope& ws) {
		// Retrieve metadata
		auto metadataPromise = src.metadataRequest().send();
		auto capTablePromise = src.capTableRequest().send();
		
		auto metadata = metadataPromise.wait(ws);
		
		Own<LocalDataStore::Entry> entry;
		
		// Check if we have the ID already
		{
			auto lStore = backingStore.lockExclusive();
			KJ_IF_MAYBE(lStore.find(metadata.getId()), rowPtr) {
				entry = rowPtr -> addRef();
			}
		}
		
		// If id not found, we need to download
		if(entry == nullptr) {
			auto rawBytes = src.rawBytesRequest().send().wait(ws);
			
			// Lock the store again, check for concurrent download and return row
			{
				auto lStore = backingStore.lockExclusive();
				
				KJ_IF_MAYBE(lStore.find(metadata.getId()), rowPtr) {
					// If found now, discard current download
					entry = rowPtr -> addRef();
				} else {
					// If not found, store row
					lStore.insert(entry -> addRef());
				}
			}
		}
		
		// Now we need to process the capability table
		auto capTable = capTablePromise.wait();
		
		auto capHooks = kj::heapArray<Maybe<Own<kj::ClientHook>>> rawCapTable(capTable.size());
		for(size_t i = 0; i < capTable.size(); ++i) {
			auto hookPtr = kj::ClientHook::from(capTable[i]);
			
			if(hookPtr != nullptr)
				capHooks[i] = hookPtr;
		}
		
		auto capClients = kj::heapArray<capnp::Capability::Client>(capTable.size());
		for(size_t i = 0; i < capTable.size(); ++i) {
			capClients[i] = capTable[i];
		}
		
		// Initialize the backend struct with everything
		LocalDataRefImpl backend;
		backend.readerTable = kj::heap<capnp::ReaderCapabilityTable>(mv(capHooks));
		backend.capTableClients = mv(capClients);
		backend.entryRef = mv(entry);
		
		backend._metadata.setRoot(metadata);
		
		// And move it into a refcounted heap instance
		Own<LocalDataRefImpl> backendRef = kj::refcounted<LocalDataRefImpl>(mv(backend));
		
		// Now construct a local data ref from the backend
		return LocalDataRef<capnp::AnyPointer(*backendRef, this -> serverSet);
	});
}

} // namespace fsc