#include <fsc/data.capnp.h>
#include <capnp/any.h>

#include "common.h"
#include "local.h"

namespace fsc {

// Internal forward declarations

namespace internal {
	class LocalDataRefImpl;
	class LocalDataServiceImpl;
}

// API forward declarations

template<typename T> class LocalDataRef;

// Unwrapper for data ref type

namespace internal {

template<typename T>
struct References_ { using Type = typename References_<capnp::FromClient<T>>::Type; };

template<typename T>
struct References_<DataRef<T>> { using Type = T; };

template<typename T>
struct References_<LocalDataRef<T>> { using Type = T; };

}

/**
 * Use this to figure out what datatype a reference points to.
 */
template<typename T>
using References = typename internal::References_<T>::Type;



// ============================================ API =============================================

/**
 * Main entry point for handling local and remote data references. Can be used to both create
 * remotely-downloadable data references with its 'publish' methods and download (as in, create
 * local copies of) remote references with its 'download' methods.
 */
class LocalDataService : public DataService::Client {
public:
	/**
	 * Downloads the data contained in the remote reference into the local backing
	 * store and links the remote capabilities into a local capability table.
	 *
	 * Returns a local data ref instance which extends the interface by DataRef
	 * with direct access to the stored data.
	 */
	template<typename Reference, typename T = References<Reference>>
	Promise<LocalDataRef<T>> download(Reference src);
	
	/**
	 * Creates a local data reference directly from a backing array and a capability table.
	 *
	 * The interpretation of the backing array depends on the seleced data type. If the
	 * data type is capnp::Data, the array is interpreted as the raw data intended to
	 * be referenced. This is e.g. intended to be used for raw memory-mapped files.
	 * Currently, any other type leads to the backing array being interpreted as containing
	 * a CapNProto message (including its segment table) with a root of the specified data
	 * type (can also be capnp::AnyPointer, capnp::AnyList or capnp::AnyStruct or similar).
	 */
	template<typename T = capnp::Data>
	LocalDataRef<T> publish(ArrayPtr<const byte> id, Array<const byte> backingArray, ArrayPtr<Maybe<Own<capnp::Capability::Client>>> capTable = kj::heapArrayBuilder<Maybe<Own<capnp::Capability::Client>>>(0).finish());
	
	/**
	 * Creates a local data reference by copying the contents of a capnproto reader.
	 * If the reader is of type capnp::Data, the byte array it points to will be copied
	 * verbatim into the backing buffer.
	 * Currently, for any other type this method will create a message containing a deepcopy
	 * copy of the data referenced by this reader and store it into the backing array (including
	 * the message's segment table).
	 */
	template<typename Reader, typename T = capnp::FromAny<Reader>>
	LocalDataRef<T> publish(ArrayPtr<const byte> id, Reader reader);
	
	/**
	 * Constructs a new data service instance using the shared backing store contained in the given
	 * library handle.
	 */
	LocalDataService(Library& lib);

	// Non-const copy constructor
	LocalDataService(LocalDataService& other);
	
	// Move constructor
	LocalDataService(LocalDataService&& other);

	// Copy assignment operator
	LocalDataService& operator=(LocalDataService& other);

	// Copy assignment operator
	LocalDataService& operator=(LocalDataService&& other);
	
	LocalDataService() = delete;
		
private:
	Own<internal::LocalDataServiceImpl> impl;
	
	LocalDataService(internal::LocalDataServiceImpl& impl);
	
	template<typename T>
	friend class LocalDataRef;
	
	friend class internal::LocalDataRefImpl;
};

/**
 * Data reference backed by local storage. In addition to the remote access functionality
 * provided by the interface in capnp::DataRef<...>, this class provides direct access to
 * locally stored data.
 * This class uses non-atomic reference counting for performance, so it can not be
 * shared across threads. To share this to other threads, pass the DataRef<...>::Client
 * capability it inherits from via RPC and use that thread's DataService to download
 * it into a local reference. If this ref and the other DataServce share the same
 * data store, the underlying data will not be copied, but shared between the references.
 */
template<typename T>
class LocalDataRef : public DataRef<T>::Client {
public:
	/**
	 * Provides direct access to the raw underlying byte array associated
	 * with this data reference.
	 */
	ArrayPtr<const byte> getRaw();
	
	/**
	 * Provides a structured view of the underlying data. If T is capnp::Data,
	 * the returned reader will be identical to getRaw(). Otherwise, this will
	 * interpret the backing array as a CapNProto message with the given type
	 * at its root. Note that if T is not capnp::Data, and the backing array
	 * can not be interpreted as a CapNProto message, this method will fail.
	 */
	typename T::Reader get();
	
	/**
	 * Provides a new data reference sharing the underling buffer and
	 * capabilities, but having a different interpretation data type.
	 */
	template<typename T2 = capnp::AnyPointer>
	class LocalDataRef<T2> as();	

	// Non-const copy constructor
	LocalDataRef(LocalDataRef<T>& other);
	
	// Move constructor
	LocalDataRef(LocalDataRef<T>&& other);

	// Copy assignment operator
	LocalDataRef<T>& operator=(LocalDataRef<T>&  other);
	
	// Move assignment operator
	LocalDataRef<T>& operator=(LocalDataRef<T>&& other);

	LocalDataRef() = delete;
	
private:
	LocalDataRef(Own<internal::LocalDataRefImpl> backend, capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>>& wrapper);

	template<typename T2>	
	LocalDataRef(LocalDataRef<T2>& other);
	
	Own<internal::LocalDataRefImpl> backend;
	
	friend class internal::LocalDataServiceImpl;
	
	template<typename T2>
	friend class LocalDataRef;
};

// ======================================== Internals ====================================


namespace internal {

/**
 * Backend implementation of the local data service.
 */
class LocalDataServiceImpl : public kj::Refcounted, public DataService::Server {
public:
	LocalDataServiceImpl(Library& h);
	Own<LocalDataServiceImpl> addRef();
	
	Promise<LocalDataRef<capnp::AnyPointer>> download(DataRef<capnp::AnyPointer>::Client src);
	LocalDataRef<capnp::AnyPointer> publish(ArrayPtr<const byte> id, Array<const byte>&& data, ArrayPtr<Maybe<Own<capnp::ClientHook>>> capTable, uint64_t cpTypeId);
	
private:
	Promise<LocalDataRef<capnp::AnyPointer>> doDownload(DataRef<capnp::AnyPointer>::Client src);
	
	capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>> serverSet;
	Library library;
	
	friend class LocalDataRefImpl;
};

/**
 * Backend implementation for locally stored data refs. Holds a reference to the
 * binary data store for the encoded binary data and a table of capabilities
 * referenced inside the binary data.
 */
class LocalDataRefImpl : public DataRef<capnp::AnyPointer>::Server, public kj::Refcounted {
public:
	using typename DataRef<capnp::AnyPointer>::Server::MetadataContext;
	using typename DataRef<capnp::AnyPointer>::Server::RawBytesContext;
	using typename DataRef<capnp::AnyPointer>::Server::CapTableContext;
	
	using Metadata = typename DataRef<capnp::AnyPointer>::Metadata;
	
	Own<LocalDataRefImpl> addRef();
	
	// Decodes the underlying data as a capnproto message
	template<typename T>
	typename T::Reader get();
	
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

	virtual ~LocalDataRefImpl() {};
	
	capnp::FlatArrayMessageReader& ensureReader();

private:
	LocalDataRefImpl() {};
	
	Maybe<capnp::FlatArrayMessageReader> maybeReader;

	friend Own<LocalDataRefImpl> kj::refcounted<LocalDataRefImpl>();
};

// Helper methods to handle the special representation for capnp::Data.

template<typename T>
typename T::Reader getDataRefAs(LocalDataRefImpl& impl);

template<>
capnp::Data::Reader getDataRefAs<capnp::Data>(LocalDataRefImpl& impl);

template<typename T>
Array<const byte> buildData(typename T::Reader reader, capnp::BuilderCapabilityTable& builderTable);

template<>
inline Array<const byte> buildData<capnp::Data>(capnp::Data::Reader reader, capnp::BuilderCapabilityTable&) {
	return kj::heapArray<const byte>(reader);
}

template<typename T>
bool checkReader(typename T::Reader, const Array<const byte>&) {
	return true;
}

template<>
inline bool checkReader<capnp::Data>(capnp::Data::Reader reader, const Array<const byte>& array) {
	return reader.begin() == array.begin();
}

template<typename T>
uint64_t constexpr capnpTypeId() { return capnp::typeId<T>(); }

template<>
inline uint64_t constexpr capnpTypeId<capnp::Data>() { return 0; }

template<>
inline uint64_t constexpr capnpTypeId<capnp::AnyPointer>() { return 1; }

template<>
inline uint64_t constexpr capnpTypeId<capnp::AnyStruct>() { return 1; }

} // namespace fsc::internal

// === class LocalDataService ===

template<typename Reader, typename T>
LocalDataRef<T> LocalDataService::publish(ArrayPtr<const byte> id, Reader data) {
	capnp::BuilderCapabilityTable capTable;
	
	Array<const byte> byteData = internal::buildData<T>(data, capTable);
			
	return impl->publish(
		id,
		mv(byteData),
		capTable.getTable(),
		internal::capnpTypeId<T>()
	).template as<T>();
}

template<typename T>
LocalDataRef<T> LocalDataService::publish(
	ArrayPtr<const byte> id,
	Array<const byte> backingArray,
	ArrayPtr<Maybe<Own<capnp::Capability::Client>>> capTable
) {
	auto hooks = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(capTable.size());
	for(auto& maybeClient : capTable) {
		KJ_IF_MAYBE(pClient, maybeClient) {
			hooks.add(capnp::ClientHook::from(**pClient));
		} else {
			hooks.add(nullptr);
		}
	}
	
	return impl->publish(
		id,
		mv(backingArray),
		hooks.finish(),
		internal::capnpTypeId<T>()
	).template as<T>();
}

template<typename Reference, typename T>
Promise<LocalDataRef<T>> LocalDataService::download(Reference src) {
	return impl -> download(src.asGeneric()).then(
		[](LocalDataRef<capnp::AnyPointer> ref) -> LocalDataRef<T> { return ref.template as<T>(); }
	);
}

// === class LocalDataRefImpl ===

template<typename T>
typename T::Reader internal::LocalDataRefImpl::get() {
	return internal::getDataRefAs<T>(*this);
}

// === class LocalDataRef ===

template<typename T>
LocalDataRef<T>::LocalDataRef(Own<internal::LocalDataRefImpl> nbackend, capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>>& wrapper) :
	capnp::Capability::Client(wrapper.add(nbackend->addRef())),
	backend(nbackend->addRef())
{}

template<typename T>
template<typename T2>	
LocalDataRef<T>::LocalDataRef(LocalDataRef<T2>& other) :
	capnp::Capability::Client((capnp::Capability::Client&) other),
	backend(other.backend -> addRef())
{}
	

template<typename T>	
LocalDataRef<T>::LocalDataRef(LocalDataRef<T>& other) :
	capnp::Capability::Client((capnp::Capability::Client&) other),
	backend(other.backend -> addRef())
{}
	

template<typename T>	
LocalDataRef<T>::LocalDataRef(LocalDataRef<T>&& other) :
	capnp::Capability::Client((capnp::Capability::Client&) other),
	backend(other.backend -> addRef())
{}

template<typename T>
LocalDataRef<T>& LocalDataRef<T>::operator=(LocalDataRef<T>& other) {
	::capnp::Capability::Client::operator=(other);
	backend = other.backend -> addRef();
	return *this;
}

template<typename T>
LocalDataRef<T>& LocalDataRef<T>::operator=(LocalDataRef<T>&& other) {
	::capnp::Capability::Client::operator=(other);
	backend = other.backend -> addRef();
	return *this;
}

template<typename T>
ArrayPtr<const byte> LocalDataRef<T>::getRaw() {
	return backend -> get<capnp::Data>();
}

template<typename T>
typename T::Reader LocalDataRef<T>::get() {
	return backend -> get<T>();
}

template<typename T>
template<typename T2>
LocalDataRef<T2> LocalDataRef<T>::as() {
	return LocalDataRef<T2>(*this);
}

// === Helper methods ===

template<typename T>
Array<const byte> internal::buildData(typename T::Reader reader, capnp::BuilderCapabilityTable& builderTable) {
	capnp::MallocMessageBuilder builder;
	
	auto root = builderTable.imbue(builder.getRoot<capnp::AnyPointer>());
	root.setAs<T>(reader);
	
	kj::Array<const capnp::word> flatArray = capnp::messageToFlatArray(builder);
	
	// Since releaseAsBytes doesn't work, we need to force the conversion
	kj::ArrayPtr<const byte> byteView(
		reinterpret_cast<const byte*>(flatArray.begin()),
		sizeof(capnp::word) * flatArray.size()
	);
	kj::Array<const byte> stableByteView = byteView.attach(kj::heap<kj::Array<const capnp::word>>(mv(flatArray)));
	return stableByteView;
}

template<typename T>
typename T::Reader internal::getDataRefAs(internal::LocalDataRefImpl& impl) {
	auto& msgReader = impl.ensureReader();
	
	// Return the reader's root at the requested type
	capnp::AnyPointer::Reader root = msgReader.getRoot<capnp::AnyPointer>();
	root = impl.readerTable -> imbue(root);
	
	// Copy root onto the heap and attach objects needed to keep it running
	return root.getAs<T>();
}

}
