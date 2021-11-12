#include <kj/array.h>

#include "data.h"

namespace fsc {
	
// class LocalDataService

LocalDataService::LocalDataService(Library& lib) :
	LocalDataService(*kj::refcounted<Impl>(lib))
{}

LocalDataService::LocalDataService(Impl& newImpl) :
	Client(newImpl.addRef()),
	impl(newImpl.addRef())
{}

DataRef<capnp::AnyPointer>::Metadata::Reader internal::LocalDataRefImpl::localMetadata() {
	return _metadata.getRoot<Metadata>();
}

Promise<void> internal::LocalDataRefImpl::metadata(MetadataContext context) {
	context.getResults().setMetadata(localMetadata());
	
	return kj::READY_NOW;
}

Promise<void> internal::LocalDataRefImpl::rawBytes(RawBytesContext context) {
	context.getResults().setData(*get<capnp::Data>());
	
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

template<>
Own<capnp::Data::Reader> internal::getDataRefAs<capnp::Data>(internal::LocalDataRefImpl& impl) {
	return kj::heap<capnp::Data::Reader>(impl.entryRef -> value.asPtr()).attach(impl.addRef());
}

// === class LocalDataService::Impl ===

LocalDataService::Impl::Impl(Library& lib) :
	library(lib -> addRef())
{}

Own<LocalDataService::Impl> LocalDataService::Impl::addRef() {
	return kj::addRef(*this);
}

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
		kj::Locked<LocalDataStore> lStore = library -> store.lockExclusive();
		KJ_IF_MAYBE(ppRow, lStore -> table.find(id)) {
			entry = (*ppRow) -> addRef();
		} else {
			entry = kj::atomicRefcounted<LocalDataStore::Entry>(id, mv(data));
		}
	}
	
	// Prepare some clients
	ArrayPtr<Maybe<Own<capnp::ClientHook>>> rawTable = capTable.getTable();
	auto clients   = kj::heapArrayBuilder<capnp::Capability::Client>    (rawTable.size());
	auto tableCopy = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(rawTable.size());
	
	for(size_t i = 0; i < rawTable.size(); ++i) {
		KJ_IF_MAYBE(hook, rawTable[i]) {
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
	metadata.setCapTableSize(rawTable.size());
	metadata.setDataSize(entry -> value.size());
		
	// Now construct a local data ref from the backend
	return LocalDataRef<capnp::AnyPointer>(*backend, this -> serverSet);
}

Promise<LocalDataRef<capnp::AnyPointer>> LocalDataService::Impl::doDownload(DataRef<capnp::AnyPointer>::Client src) {
	// Use a fiber for downloading
	
	return kj::startFiber(65536, [this, src](kj::WaitScope& ws) mutable {
		// Retrieve metadata
		auto metadataPromise = src.metadataRequest().send();
		auto capTablePromise = src.capTableRequest().send();
		
		auto metadata = metadataPromise.wait(ws).getMetadata();
		
		Own<const LocalDataStore::Entry> entry;
		
		// Check if we have the ID already
		{
			auto lStore = library -> store.lockExclusive();
			KJ_IF_MAYBE(rowPtr, lStore -> table.find(metadata.getId())) {
				entry = (*rowPtr) -> addRef();
			}
		}
		
		// If id not found, we need to download
		if(entry.get() == nullptr) {
			auto rawBytes = src.rawBytesRequest().send().wait(ws);
			
			// Lock the store again, check for concurrent download and return row
			{
				auto lStore = library -> store.lockExclusive();
				
				KJ_IF_MAYBE(rowPtr, lStore -> table.find(metadata.getId())) {
					// If found now, discard current download
					entry = (*rowPtr) -> addRef();
				} else {
					// If not found, store row
					lStore -> table.insert(entry -> addRef());
				}
			}
		}
		
		// Now we need to process the capability table
		auto capTable = capTablePromise.wait(ws).getTable();
		
		auto capHooks = kj::heapArray<Maybe<Own<capnp::ClientHook>>>(capTable.size());
		for(unsigned int i = 0; i < capTable.size(); ++i) {
			Own<capnp::ClientHook> hookPtr = capnp::ClientHook::from(capTable[i]);
			
			if(hookPtr.get() != nullptr)
				capHooks[i] = mv(hookPtr);
		}
		
		//auto capClients = kj::heapArray<capnp::Capability::Client>(capTable.size());
		auto capClients = kj::heapArrayBuilder<capnp::Capability::Client>(capTable.size());
		for(unsigned int i = 0; i < capTable.size(); ++i) {
			capClients.add(capTable[i]);
		}
		
		// Initialize the backend struct with everything
		auto backend = kj::refcounted<internal::LocalDataRefImpl>();
		backend->readerTable = kj::heap<capnp::ReaderCapabilityTable>(mv(capHooks));
		backend->capTableClients = capClients.finish();
		backend->entryRef = mv(entry);
		
		backend->_metadata.setRoot(metadata);
				
		// Now construct a local data ref from the backend
		return LocalDataRef<capnp::AnyPointer>(*backend, this -> serverSet);
	});
}

} // namespace fsc