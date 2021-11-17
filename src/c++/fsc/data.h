#include <fsc/data.capnp.h>
#include <capnp/any.h>

#include "common.h"
#include "local.h"

namespace fsc {

// Internal forward declarations

namespace internal {
	class LocalDataRefImpl;
}

// API forward declarations

template<typename T> class LocalDataRef;

// Unwrapper for data ref type

template<typename T>
struct References_ { using Type = typename References_<capnp::FromClient<T>>::Type; };

template<typename T>
struct References_<DataRef<T>> { using Type = T; };

template<typename T>
struct References_<LocalDataRef<T>> { using Type = T; };

template<typename T>
using References = typename References_<T>::Type;



// ============================================ API =============================================

class LocalDataService : public DataService::Client {
public:
	template<typename Reference, typename T = References<Reference>>
	Promise<LocalDataRef<T>> download(Reference src);
	
	template<typename T = capnp::Data>
	LocalDataRef<T> publish(ArrayPtr<const byte> id, Array<const byte> backingArray, ArrayPtr<Maybe<capnp::Capability::Client>> capTable = kj::heapArray<Maybe<capnp::Capability::Client>>({}));
	
	template<typename Reader, typename T = capnp::FromReader<Reader>>
	LocalDataRef<T> publish(ArrayPtr<const byte> id, Reader reader);
	
	
	/*LocalDataRef<capnp::Data> publish(ArrayPtr<const byte> id, Array<const byte>&& data);
	LocalDataRef<capnp::Data> publish(ArrayPtr<const byte> id, capnp::Data::Reader);
	
	template<typename T>
	LocalDataRef<T> publish(ArrayPtr<const byte> id, typename T::Reader data);*/
	
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
	ArrayPtr<const byte> getRaw();
	
	typename T::Reader get();
	
	template<typename T2 = capnp::AnyPointer>
	class LocalDataRef<T2> as();	

	LocalDataRef(LocalDataRef<T>& other);
	LocalDataRef(LocalDataRef<T>&& other);

	LocalDataRef<T>& operator=(LocalDataRef<T>&  other);
	LocalDataRef<T>& operator=(LocalDataRef<T>&& other);

	LocalDataRef() = delete;
	
private:
	LocalDataRef(Own<internal::LocalDataRefImpl> backend, capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>>& wrapper);

	template<typename T2>	
	LocalDataRef(LocalDataRef<T2>& other);
	
	Own<internal::LocalDataRefImpl> backend;
	
	friend class LocalDataService::Impl;
	
	template<typename T2>
	friend class LocalDataRef;
};

// ======================================== Internals ====================================

/**
 * Backend implementation of the local data service.
 */
class LocalDataService::Impl : public kj::Refcounted, public DataService::Server {
public:
	Impl(Library& h);
	Own<Impl> addRef();
	
	Promise<LocalDataRef<capnp::AnyPointer>> download(DataRef<capnp::AnyPointer>::Client src);
	LocalDataRef<capnp::AnyPointer> publish(ArrayPtr<const byte> id, Array<const byte>&& data, ArrayPtr<Maybe<Own<capnp::ClientHook>>> capTable, uint64_t cpTypeId);
	
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
Array<const byte> buildData<capnp::Data>(capnp::Data::Reader reader, capnp::BuilderCapabilityTable&) {
	return kj::heapArray<const byte>(reader);
}

template<typename T>
bool checkReader(typename T::Reader, const Array<const byte>&) {
	return true;
}

template<>
bool checkReader<capnp::Data>(capnp::Data::Reader reader, const Array<const byte>& array) {
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
	ArrayPtr<Maybe<capnp::Capability::Client>> capTable
) {
	auto hooks = kj::heapArrayBuilder<Maybe<Own<capnp::ClientHook>>>(capTable.size());
	for(auto& maybeClient : capTable) {
		KJ_IF_MAYBE(pClient, maybeClient) {
			hooks.add(capnp::ClientHook::from(*pClient));
		} else {
			hooks.add(nullptr);
		}
	}
	
	return impl->publish(
		id,
		backingArray,
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
