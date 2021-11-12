#include <fsc/data.capnp.h>
#include <capnp/any.h>

#include "common.h"
#include "local.h"

namespace fsc {

// Internal forward declarations

namespace internal {
	class LocalDataRefImpl;
}

template<typename T> class LocalDataRef;

// API

class LocalDataService : public DataService::Client {
public:
	template<typename T>
	Promise<LocalDataRef<T>> download(typename DataRef<T>::Client src);
	
	LocalDataRef<void> publish(Array<byte> id, Array<byte>&& data);
	
	template<typename T>
	LocalDataRef<capnp::Data> publish(Array<byte> id, typename T::Reader data);
	
	LocalDataService(Library& lib);
	
private:
	class Impl;
	Own<Impl> impl;
	
	LocalDataService(Impl& impl);
	
	template<typename T>
	friend class LocalDataRef;
	
	friend class internal::LocalDataRefImpl;
};

/**
 * DataRef backed by local storage.
 */
template<typename T>
class LocalDataRef : public DataRef<T>::Client {
public:
	ArrayPtr<byte> getRaw();
	
	Own<typename T::Reader> get();
	
	template<typename T2 = capnp::AnyPointer>
	class LocalDataRef<T2> as();
	
private:
	LocalDataRef(internal::LocalDataRefImpl& backend, capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>>& wrapper);	
	Own<internal::LocalDataRefImpl> backend;
	
	friend class LocalDataService::Impl;
};

// Internal implementation

/**
 * Backend implementation of the local data service.
 */
class LocalDataService::Impl : public kj::Refcounted, public DataService::Server {
public:
	Impl(Library& h);
	Own<Impl> addRef();
	
	Promise<LocalDataRef<capnp::AnyPointer>> download(DataRef<capnp::AnyPointer>::Client src);
	LocalDataRef<capnp::AnyPointer> publish(Array<byte> id, Array<byte>&& data, capnp::BuilderCapabilityTable&& capTable, uint64_t cpTypeId);
	
private:
	Promise<LocalDataRef<capnp::AnyPointer>> doDownload(DataRef<capnp::AnyPointer>::Client src);
	
	capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>> serverSet;
	Library library;
	
	friend class internal::LocalDataRefImpl;
};


namespace internal {

/**
 * Backend implementation for locally stored data refs. Holds a reference to the
 * binary data store for the encoded binary data and a table of capabilities
 * referenced inside the binary data.
 */
class LocalDataRefImpl : public kj::Refcounted, public DataRef<capnp::AnyPointer>::Server {
public:
	using typename DataRef<capnp::AnyPointer>::Server::MetadataContext;
	using typename DataRef<capnp::AnyPointer>::Server::RawBytesContext;
	using typename DataRef<capnp::AnyPointer>::Server::CapTableContext;
	
	using Metadata = typename DataRef<capnp::AnyPointer>::Metadata;
	
	LocalDataRefImpl(LocalDataRefImpl&&) = default;
	
	Own<LocalDataRefImpl> addRef();
	
	// Decodes the underlying data as a capnproto message
	template<typename T>
	Own<typename T::Reader> get();
	
	// capnp::Data is encoded using raw bytes
	// Therefore, the get() method has to be specialized for this type
	template<>
	Own<capnp::Data::Reader> get<capnp::Data>();
	
	// Returns a reader to the locally stored metadata
	Metadata::Reader localMetadata();
	
	Promise<void> metadata(MetadataContext) override ;
	Promise<void> rawBytes(RawBytesContext) override ;
	Promise<void> capTable(CapTableContext) override ;
	
	// Reference to the local data store entry holding our data
	Own<const LocalDataStore::Entry> entryRef;
	
	// Array-of-clients view onto the capability table
	Array<capnp::Capability::Client> capTableClients;
	
	// ReaderCapabilityTable view onto the capability table
	Own<capnp::ReaderCapabilityTable> readerTable;
	
	// Serialized metadata
	capnp::MallocMessageBuilder _metadata;

private:
	LocalDataRefImpl() {};
	friend Own<LocalDataRefImpl> kj::refcounted<LocalDataRefImpl>();
};

template<typename T>
typename Own<typename T::Reader> getDataRefAs(LocalDataRefImpl& impl);

template<>
Own<capnp::Data::Reader> getDataRefAs<capnp::Data>(LocalDataRefImpl& impl);

} // namespace fsc::internal

// Inline implementation

// === class LocalDataService::Impl ===

// === class LocalDataRefImpl ===

template<typename T>
Own<typename T::Reader> internal::getDataRefAs(internal::LocalDataRefImpl& impl) {
	ArrayPtr<byte> bytePtr = *getDataRefAs<capnp::Data>(impl);
	ArrayPtr<word> wordPtr = reinterpret_cast<ArrayPtr<word>>(bytePtr);
	
	auto msgReader = kj::heap<capnp::FlatArrayMessageReader>(wordPtr);
	
	T2::Reader root = msgReader.getRoot<T2>();
	root = impl.readerTable -> imbue(root);
	
	return kj::heap<T2::Reader>(root).attach(mv(msgReader)).attach(impl.addRef());
}

// === class LocalDataRef ===

template<typename T>
LocalDataRef<T>::LocalDataRef(internal::LocalDataRefImpl& backend, capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>>& wrapper) :
	Client(wrapper.add(backend.addRef())),
	backend(backend.addRef())
{}

template<typename T>
ArrayPtr<byte> LocalDataRef<T>::getRaw() {
	return backend -> get<capnp::Data>();
}

template<typename T>
Own<typename T::Reader> LocalDataRef<T>::get() {
	return backend -> get<T>();
}

template<typename T>
template<typename T2>
LocalDataRef<T2> LocalDataRef<T>::as() {
	return LocalDataRef<T2>(backend);
}

}