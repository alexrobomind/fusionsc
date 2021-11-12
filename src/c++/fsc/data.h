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
	Array<const byte> getRaw();
	
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
	
	Own<LocalDataRefImpl> addRef();
	
	// Decodes the underlying data as a capnproto message
	template<typename T>
	Own<typename T::Reader> get();
	
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
Own<typename T::Reader> getDataRefAs(LocalDataRefImpl& impl);

template<>
Own<capnp::Data::Reader> getDataRefAs<capnp::Data>(LocalDataRefImpl& impl);

} // namespace fsc::internal

// Inline implementation

// === class LocalDataService::Impl ===

// === class LocalDataRefImpl ===

template<typename T>
Own<typename T::Reader> internal::LocalDataRefImpl::get() {
	return internal::getDataRefAs<T>(*this);
}

template<typename T>
Own<typename T::Reader> internal::getDataRefAs(internal::LocalDataRefImpl& impl) {
	// Obtain data as a byte pointer (note that this drops all attached objects to keep alive0
	ArrayPtr<const byte> bytePtr = *getDataRefAs<capnp::Data>(impl);
	
	// Cast the data to a word array (let's hope they are aligned properly)
	ArrayPtr<const capnp::word> wordPtr = ArrayPtr<const capnp::word>(
		reinterpret_cast<const capnp::word*>(bytePtr.begin()),
		bytePtr.size() / sizeof(capnp::word)
	);
	
	// Construct a message reader over the array
	auto msgReader = kj::heap<capnp::FlatArrayMessageReader>(wordPtr);
	
	// Return the reader's root at the requested type
	typename T::Reader root = msgReader -> getRoot<T>();
	root = impl.readerTable -> imbue(root);
	
	// Copy root onto the heap and attach objects needed to keep it running
	return kj::heap<typename T::Reader>(root).attach(mv(msgReader)).attach(impl.addRef());
}

// === class LocalDataRef ===

template<typename T>
LocalDataRef<T>::LocalDataRef(internal::LocalDataRefImpl& backend, capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>>& wrapper) :
	DataRef<T>::Client(wrapper.add(backend.addRef())),
	backend(backend.addRef())
{}

template<typename T>
Array<const byte> LocalDataRef<T>::getRaw() {
	Own<capnp::Data::Reader> reader = backend -> get<capnp::Data>();
	
	// The below operation converts the arrayptr view of the reader
	// into a refcount owning array.
	ArrayPtr<const byte> byteView = *reader;
	return byteView.attach(reader);
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