#include <kj/map.h>

#include <botan/hash.h>

namespace fsc {

namespace internal {

// Specializations for TensorFor

#define FSC_DECLARE_TENSOR(Val, T) \
	template<> \
	struct TensorFor_<Val> { using Type = T; }

FSC_DECLARE_TENSOR(uint32_t, UInt32Tensor);
FSC_DECLARE_TENSOR( int32_t,  Int32Tensor);
FSC_DECLARE_TENSOR(uint64_t, UInt64Tensor);
FSC_DECLARE_TENSOR( int64_t,  Int32Tensor);
FSC_DECLARE_TENSOR(float,  Float32Tensor);
FSC_DECLARE_TENSOR(double,  Float64Tensor);

#undef FSC_DECLARE_TENSOR

struct MMapTemporary {
	int64_t dedicatedObjectSize = 1024 * 1024;
	int64_t fileSize = 1024 * 1024 * 20;
	
	inline kj::Array<byte> request(size_t size) {
		KJ_REQUIRE(fileSize >= dedicatedObjectSize);
		
		if(size >= dedicatedObjectSize) {
			// Otherwise, if the object is big, give it is own file
			auto mapping = dir->createTemporary()->mmapWritable(0, size);
			
			auto ptr = mapping->get();
			return ptr.attach(mv(mapping));
		} 
		
		// Check if we can stuff the object into the remainder of this file
		if(offset + size >= fileSize) {
			reset();
		}
		
		return alloc(size);
	};
	
	inline MMapTemporary(Own<const kj::Directory> dir) : dir(mv(dir)) {
		reset();
	}
	
private:
	inline void reset() {
		offset = 0;
		file = dir->createTemporary();
	}
	
	kj::Array<byte> alloc(uint64_t size) {
		auto mapping = file->mmapWritable(offset, size);
		offset += size;
		
		auto ptr = mapping->get();
		return ptr.attach(mv(mapping));
	}		
	
	uint64_t offset = 0;
	Own<const kj::File> file;
	Own<const kj::Directory> dir;
};

/**
 * Backend implementation of the local data service.
 */
class LocalDataServiceImpl : public kj::Refcounted, public DataService::Server {
public:
	using Nursery = LocalDataService::Nursery;
	
	LocalDataServiceImpl(Library& h);
	Own<LocalDataServiceImpl> addRef();
	
	Promise<LocalDataRef<capnp::AnyPointer>> download(DataRef<capnp::AnyPointer>::Client src, bool recursive);
	
	LocalDataRef<capnp::AnyPointer> publish(DataRef<T>::Metadata::Reader metaData, Array<const byte>&& data, ArrayPtr<Maybe<Own<capnp::ClientHook>>> capTable);
	
	Promise<void> buildArchive(DataRef<capnp::AnyPointer>::Client ref, Archive::Builder out, Maybe<Nursery&> nursery);
	Promise<void> writeArchive(DataRef<capnp::AnyPointer>::Client ref, const kj::File& out);
	LocalDataRef<capnp::AnyPointer> publishArchive(Archive::Reader archive);
	LocalDataRef<capnp::AnyPointer> publishArchive(const kj::ReadableFile& f, const capnp::ReaderOptions options);
	
	kj::FiberPool downloadPool;
		
	Promise<void> clone(CloneContext context) override;
	Promise<void> store(StoreContext context) override;
	
	inline void setLimits(LocalDataService::Limits newLimits);
	
	void setChunkDebugMode();
	
private:
	Promise<LocalDataRef<capnp::AnyPointer>> doDownload(DataRef<capnp::AnyPointer>::Client src, bool recursive);
	Promise<Archive::Entry::Builder> createArchiveEntry(DataRef<capnp::AnyPointer>::Client ref, kj::TreeMap<ID, capnp::Orphan<Archive::Entry>>& entries, capnp::Orphanage orphanage, Maybe<Nursery&> nursery);
	
	capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>> serverSet;
	Library library;
	
	LocalDataService::Limits limits;
	MMapTemporary fileBackedMemory;
	
	friend class LocalDataRefImpl;
	
	bool debugChunks = false;
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
	typename T::Reader get(const capnp::ReaderOptions& options);
	
	// Returns a reader to the locally stored metadata
	Metadata::Reader localMetadata();
	
	Promise<void> metadata(MetadataContext) override ;
	Promise<void> rawBytes(RawBytesContext) override ;
	Promise<void> capTable(CapTableContext) override ;
	Promise<void> transmit(TransmitContext) override ;
	
	// Reference to the local data store entry holding our data
	Own<const LocalDataStore::Entry> entryRef;
	
	// Array-of-clients view onto the capability table
	Array<capnp::Capability::Client> capTableClients;
	
	// ReaderCapabilityTable view onto the capability table
	Own<capnp::ReaderCapabilityTable> readerTable;
	
	// Serialized metadata
	capnp::MallocMessageBuilder _metadata;

	virtual ~LocalDataRefImpl() {};
	
	capnp::FlatArrayMessageReader& ensureReader(const capnp::ReaderOptions& options);

private:
	LocalDataRefImpl() {};
	
	Maybe<capnp::FlatArrayMessageReader> maybeReader;

	friend Own<LocalDataRefImpl> kj::refcounted<LocalDataRefImpl>();
};

// Helper methods to handle the special representation for capnp::Data.

template<typename T>
typename T::Reader getDataRefAs(LocalDataRefImpl& impl, const capnp::ReaderOptions& options);

template<>
capnp::Data::Reader getDataRefAs<capnp::Data>(LocalDataRefImpl& impl, const capnp::ReaderOptions& options);

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
uint64_t capnpTypeId(typename T::Reader reader) { return capnp::typeId<T>(); }

template<>
inline uint64_t capnpTypeId<capnp::Data>(capnp::Data::Reader reader) { return 0; }

template<>
inline uint64_t capnpTypeId<capnp::AnyPointer>(capnp::AnyPointer::Reader reader) { return 1; }

template<>
inline uint64_t capnpTypeId<capnp::AnyStruct>(capnp::AnyStruct::Reader reader) { return 1; }

template<>
inline uint64_t capnpTypeId<capnp::DynamicStruct>(capnp::DynamicStruct::Reader reader) { return reader.getSchema().getProto().getId(); }
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
		internal::capnpTypeId<T>(data)
	).template as<T>();
}

template<typename Reader, typename IDReader, typename T, typename T2>
Promise<LocalDataRef<T>> LocalDataService::publish(IDReader dataForID, Reader data, kj::StringPtr hashFunction) {
	Promise<ID> id = ID::fromReaderWithRefs(dataForID);
	
	return id.then([this, data, dataForID, hashFunction = kj::heapString(hashFunction)](ID id) {
		auto hash = Botan::HashFunction::create(hashFunction.cStr());
		
		hash->update_le(internal::capnpTypeId<capnp::FromAny<IDReader>>(dataForID));
		hash->update(id.data.begin(), id.data.size());
		
		auto newId = kj::heapArray<byte>(hash->output_length());
		hash->final(newId.begin());
		
		return publish(newId, data);
	});
}

template<typename T>
LocalDataRef<T> LocalDataService::publish(
	DataRef<T>::Metadata::Reader metaData,
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
		metaData.asDataRefGeneric(),
		mv(backingArray),
		hooks.finish(),
	).template as<T>();
}

template<typename Reference, typename T>
Promise<LocalDataRef<T>> LocalDataService::download(Reference src, bool recursive) {
	return impl -> download(src.asGeneric(), recursive).then(
		[](LocalDataRef<capnp::AnyPointer> ref) -> LocalDataRef<T> { return ref.template as<T>(); }
	);
}

template<typename Reference, typename T>
Promise<Maybe<LocalDataRef<T>>> LocalDataService::downloadIfNotNull(Reference src, bool recursive) {
	bool isNull = capnp::ClientHook::from(cp(src))->isNull();
	
	if(isNull) {
		Maybe<LocalDataRef<T>> result = nullptr;
		return result;
	}
	
	return download(src, recursive)
	.then([](LocalDataRef<T> result) mutable -> Maybe<LocalDataRef<T>> {
		return result;
	});
}

template<typename Ref, typename T>
Promise<void> LocalDataService::buildArchive(Ref ref, Archive::Builder out, Maybe<Nursery&> nursery) {
	return impl -> buildArchive(ref.asGeneric(), out, nursery);
}

template<typename Ref, typename T>
Promise<void> LocalDataService::writeArchive(Ref ref, const kj::File& out) {
	return impl -> writeArchive(ref.asGeneric(), out);
}

template<typename T>
LocalDataRef<T> LocalDataService::publishArchive(Archive::Reader in) {
	return impl -> publishArchive(in).as<T>();
}

template<typename T>
LocalDataRef<T> LocalDataService::publishArchive(const kj::ReadableFile& in, const capnp::ReaderOptions options) {
	return impl -> publishArchive(in, options).as<T>();
}

// === class LocalDataRefImpl ===

template<typename T>
typename T::Reader internal::LocalDataRefImpl::get(const capnp::ReaderOptions& options) {
	return internal::getDataRefAs<T>(*this, options);
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
	return backend -> get<capnp::Data>(READ_UNLIMITED);
}

template<typename T>
typename T::Reader LocalDataRef<T>::get(const capnp::ReaderOptions& options) {
	return backend -> get<T>(options);
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

// === function attachToClient ===

namespace internal {
	kj::Own<capnp::Capability::Server> createProxy(capnp::Capability::Client client);
}

template<typename T, typename Cap, typename... Attachments>
typename Cap::Client attach(T src, Attachments&&... attachments) {
	capnp::Capability::Client typelessClient(mv(src));
	auto proxy = internal::createProxy(mv(typelessClient));
	
	proxy = proxy.attach(fwd<Attachments>(attachments)...);
	return capnp::Capability::Client(mv(proxy)).castAs<Cap>();
}

// === struct Cache ===

template<typename Key, typename T, template<typename, typename> typename Map>
struct Cache<Key, T, Map>::Holder : public kj::Refcounted {
	Own<T> target;
	size_t refcount = 0;
	
	void clear() {
		target = nullptr;
	}
	
	Own<Holder> addRef() { return kj::addRef(*this); }
};

template<typename Key, typename T, template<typename, typename> typename Map>
struct Cache<Key, T, Map>::Ref {
	Own<Holder> target;
	
	Ref() : target(nullptr)
	{}
	
	Ref(Holder& newTarget) : target(newTarget.addRef()) {
		acquire();
	}
	
	Ref(Ref& other) : target(other.target->addRef()) {
		acquire();
	}
	
	Ref(Ref&& other) : target(mv(other.target)) {
	}
	
	Ref& operator=(Ref& other) {
		// Already set
		if(other.target == target) {
			return *this;
		}
		
		release();
		target = other.target;
		acquire();
		
		return *this;
	}
	
	Ref& operator=(Ref&& other) {
		release();
		target = other.target;
		other.target = nullptr;
		
		return *this;
	}	
	
	void acquire() {
		if(target.get()== nullptr)
			return;
		
		++(target->refcount);
	}
	
	void release() {
		if(target.get() == nullptr)
			return;
		
		if(--(target->refcount) == 0)
			target->clear();
	}
	
	~Ref() {
		release();
	}
};

template<typename Key, typename T, template<typename, typename> typename Map>
typename Cache<Key, T, Map>::InsertResult Cache<Key, T, Map>::insert(Key key, T t) {
	Own<Holder>& pHolder = map.findOrCreate(
		key,
		[&key]() -> typename Map<Key, Own<Holder>>::Entry { return { key, kj::refcounted<Holder>() }; }
	);
	
	if(pHolder->target.get() == nullptr)
		pHolder->target = kj::heap<T>(mv(t));

	return { *(pHolder->target), Ref(*pHolder) };
}

template<typename Key, typename T, template<typename, typename> typename Map>
Maybe<T&> Cache<Key, T, Map>::find(Key key) {
	KJ_IF_MAYBE(pHolder, map.find(key)) {
		if((**pHolder).target.get() == nullptr)
			return nullptr;
		
		return *((**pHolder).target);
	}
	
	return nullptr;
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
typename T::Reader internal::getDataRefAs(internal::LocalDataRefImpl& impl, const capnp::ReaderOptions& options) {
	auto& msgReader = impl.ensureReader(options);
	
	// Return the reader's root at the requested type
	capnp::AnyPointer::Reader root = msgReader.getRoot<capnp::AnyPointer>();
	root = impl.readerTable -> imbue(root);
	
	// Copy root onto the heap and attach objects needed to keep it running
	return root.getAs<T>();
}

}