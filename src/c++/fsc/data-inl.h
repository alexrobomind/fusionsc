namespace fsc {

namespace internal {

/**
 * Backend implementation of the local data service.
 */
class LocalDataServiceImpl : public kj::Refcounted, public DataService::Server {
public:
	LocalDataServiceImpl(Library& h);
	Own<LocalDataServiceImpl> addRef();
	
	Promise<LocalDataRef<capnp::AnyPointer>> download(DataRef<capnp::AnyPointer>::Client src, bool recursive);
	LocalDataRef<capnp::AnyPointer> publish(ArrayPtr<const byte> id, Array<const byte>&& data, ArrayPtr<Maybe<Own<capnp::ClientHook>>> capTable, uint64_t cpTypeId);
	
	Promise<void> buildArchive(DataRef<capnp::AnyPointer>::Client ref, Archive::Builder out);
	Promise<void> writeArchive(DataRef<capnp::AnyPointer>::Client ref, kj::File& out);
	LocalDataRef<capnp::AnyPointer> publishArchive(Archive::Reader archive);
	
	kj::FiberPool downloadPool;
	
private:
	Promise<LocalDataRef<capnp::AnyPointer>> doDownload(DataRef<capnp::AnyPointer>::Client src, bool recursive);
	
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

// Helper methods to determine type ids.

template<typename T>
uint64_t constexpr capnpTypeId() { return capnp::typeId<T>(); }

template<>
inline uint64_t constexpr capnpTypeId<capnp::Data>() { return 0; }

template<>
inline uint64_t constexpr capnpTypeId<capnp::AnyPointer>() { return 1; }

template<>
inline uint64_t constexpr capnpTypeId<capnp::AnyStruct>() { return 1; }

// Helper methods to read / write data in tensors
	
template<typename T, typename Val = typename capnp::ListElementType<capnp::FromAny<T>>>
T tensorGetFast(const T& data, const capnp::List<uint64_t>::Reader& shape, const ArrayPtr<size_t> index);

inline size_t linearIndex(const capnp::List<uint64_t>::Reader& shape, const ArrayPtr<size_t> index);

// Type inference helper that tells what a data ref references

template<typename T>
struct References_ { using Type = typename References_<capnp::FromClient<T>>::Type; };

template<typename T>
struct References_<DataRef<T>> { using Type = T; };

template<typename T>
struct References_<LocalDataRef<T>> { using Type = T; };

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
Promise<LocalDataRef<T>> LocalDataService::download(Reference src, bool recursive) {
	return impl -> download(src.asGeneric(), recursive).then(
		[](LocalDataRef<capnp::AnyPointer> ref) -> LocalDataRef<T> { return ref.template as<T>(); }
	);
}

template<typename Ref, typename T>
Promise<void> LocalDataService::buildArchive(Ref ref, Archive::Builder out) {
	return impl -> buildArchive(ref.asGeneric(), out);
}

template<typename T>
LocalDataRef<T> LocalDataService::publishArchive(Archive::Reader in) {
	return impl -> publishArchive(in).as<T>();
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

template<typename T>
ArrayPtr<const byte> LocalDataRef<T>::getID() {
	return backend -> localMetadata().getId();
}

template<typename T>
ArrayPtr<capnp::Capability::Client> LocalDataRef<T>::getCapTable() {
	return backend -> capTableClients.asPtr();
}

template<typename T>
uint64_t LocalDataRef<T>::getTypeID() {
	return backend -> localMetadata().getTypeId();
}

// === class TensorAccessor ===

template<typename T>
TensorVal<T> TensorReader<T>::get(const ArrayPtr<size_t> index) {
	return internal::tensorGetFast(data, shape, index);
}

template<typename T>
TensorReader<T>::TensorReader(const T ref) :
	data(ref.getData()),
	shape(ref.getShape())
{}

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

inline size_t internal::linearIndex(const capnp::List<uint64_t>::Reader& shape, const ArrayPtr<size_t> index) {
	size_t linearIndex = 0;
	size_t stride = 1;
	for(int dim = (int) index.size() - 1; dim >= 0; --dim) {
		linearIndex += index[dim] * stride;
		stride *= shape[dim];
	}
	
	return linearIndex;
}

template<typename T, typename Val>
T internal::tensorGetFast(const T& data, const capnp::List<uint64_t>::Reader& shape, const ArrayPtr<size_t> index) {
	return data[linearIndex(shape, index)];
}

template<typename T>
TensorVal<T> tensorGet(const T& tensor, const ArrayPtr<size_t> index) {
	return tensor.getData()[internal::linearIndex(tensor.getShape(), index)];
}

template<typename T>
TensorVal<T> tensorSet(const T& tensor, const ArrayPtr<size_t> index, TensorVal<T> value) {
	return tensor.getData().set(internal::linearIndex(tensor.getShape(), index), value);
}

}