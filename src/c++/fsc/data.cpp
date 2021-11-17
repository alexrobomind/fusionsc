#include <kj/array.h>
#include <capnp/generated-header-support.h>
#include <capnp/serialize.h>

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
	library(lib -> addRef())
{}

Own<internal::LocalDataServiceImpl> internal::LocalDataServiceImpl::addRef() {
	return kj::addRef(*this);
}

Promise<LocalDataRef<capnp::AnyPointer>> internal::LocalDataServiceImpl::download(DataRef<capnp::AnyPointer>::Client src) {
	// Check if the capability is actually local
	return serverSet.getLocalServer(src).then([src, this](Maybe<DataRef<capnp::AnyPointer>::Server&> maybeServer) mutable -> Promise<LocalDataRef<capnp::AnyPointer>> {
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
			return doDownload(src);
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

Promise<LocalDataRef<capnp::AnyPointer>> internal::LocalDataServiceImpl::doDownload(DataRef<capnp::AnyPointer>::Client src) {
	// Use a fiber for downloading
	
	return kj::startFiber(65536, [this, src](kj::WaitScope& ws) mutable {
		// Retrieve metadata and cap table
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
		return LocalDataRef<capnp::AnyPointer>(backend->addRef(), this -> serverSet);
	});
}

} // namespace fsc
