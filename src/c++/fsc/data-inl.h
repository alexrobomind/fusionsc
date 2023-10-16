#include <kj/map.h>

#include <botan/hash.h>

#include <unordered_map>

#include <capnp/serialize.h>

#include "typing.h"
#include "memory.h"

namespace fsc {

struct DBCache;

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

	
/** Common download logic.
 * 
 * In the implementation of this library, a frequent step is the transmission process required
 * to locally transfer / store DataRefs. This is a fairly complex process to remain efficient. This
 * class manages the organization of the process, while subclasses are required to implement the actual
 * steps.
 *
 * The protocol for the download process is a follows:
 *
 *  1. Check whether the target DataRef is already locally materialized (e.g. a local data ref). This
 *     should only rely on fast processes not requiring network communication, and therefore not call
 *     any methods on the ref directly. Instead, it can inspect the resolved ref and check whether its
 *     backend implementation is known (e.g. through CapabilityServerSet). This step is implemented in
 *     unwrap().
 *
 *  2. Optionally register a download process for the target DataRef. This allows the transfer of recursive
 *     structures and optimizes graph transfers with a lot of sharing.
 *
 *  3. Query the DataRef for its metadata and references to other capabilities. After receiving the references,
 *     adjustRef() is called for each of them to potentially kick off child downloads etc.
 *
 *  4. Check whether the DataRef can be constructed from locally cached data, based on the given metadata (
 *     which include a data hash).
 *
 *  5. Perform a piecewise streaming transfer of the data from the remote reference. This entails hashing
 *     the data as they are streamed in and, after transfer, storing the adjusted hash in the metadata.
 *
 *  6. Build the final result type from metadata, refs, and the stored data.
 */
template<typename Result>
struct DownloadTask : public kj::Refcounted {
	using ResultType = Result;
	
	struct Registry : public kj::Refcounted {
		std::unordered_map<capnp::ClientHook*, DownloadTask<Result>*> activeDownloads;
		
		Own<Registry> addRef() { return kj::addRef(*this); }
	};
	
	struct Context {
		mutable Own<Registry> registry;
		
		Context() : registry(kj::refcounted<Registry>()) {}
		Context(const Context& other) : registry(other.registry -> addRef()) {}
		Context(Context&& other) = default;
		
		Context& operator=(const Context& other) { registry = other.registry -> addRef(); }
		Context& operator=(Context&& other) = default;
	};
	
	DownloadTask(DataRef<capnp::AnyPointer>::Client src, Context context);
	virtual ~DownloadTask();
	
	//! Check whether "src" can be directly unwrapped
	virtual Promise<Maybe<Result>> unwrap() { return Maybe<Result>(nullptr); }
	
	//! Adjust refs e.g. by performing additional downloads. If the resulting client is broken with an exception of type "unimplemented", the original ref is used instead.
	virtual capnp::Capability::Client adjustRef(capnp::Capability::Client ref) { return ref; }
	
	//! Check whether we can build a result from given metadata and captable
	virtual Promise<Maybe<Result>> useCached() { return Maybe<Result>(nullptr); }
	
	virtual Promise<void> beginDownload() { return READY_NOW; }
	virtual Promise<void> receiveData(kj::ArrayPtr<const byte> data) = 0;
	virtual Promise<void> finishDownload() { return READY_NOW; }
	
	virtual Promise<Result> buildResult() = 0;
	
	Own<DownloadTask<Result>> addRef() { return kj::addRef(*this); }
	
	Promise<Result> output() { return result.addBranch().attach(addRef()); }
	
	//! The original DataRef under download
	DataRef<capnp::AnyPointer>::Client src;
	
	Temporary<DataRefMetadata> metadata;
	kj::Array<capnp::Capability::Client> capTable;
	
	Context ctx;
	
private:
	Promise<Result> actualTask();
	
	//! Access to the final result of the download task. To enable sharing, this must support copy assignment or reference counting
	ForkedPromise<Result> result;
	
	/* //! Sub-task for metadata transfer
	Promise<void> downloadMetadata();
	
	//! Sub-task for reference cap table download and adjustment
	Promise<void> downloadCapTable();*/
	Promise<void> downloadMetaAndCapTable();
	
	//! Sub-task for data transfer and hashing
	Promise<void> downloadData();
	
	//! Pre-check for already-local refs and ongoing downloads
	Promise<Maybe<Result>> checkLocalAndRegister();
	
	struct TransmissionReceiver : public DataRef<capnp::AnyPointer>::Receiver::Server {
		kj::ListLink<TransmissionReceiver> listLink;
		Maybe<DownloadTask&> parent;
		
		TransmissionReceiver(DownloadTask<Result>& parent) ;
		~TransmissionReceiver();
		
		Promise<void> begin(BeginContext ctx) override;
		Promise<void> receive(ReceiveContext ctx) override;
		Promise<void> done(DoneContext ctx) override;
		
		void clear();
	};
	
	std::unique_ptr<Botan::HashFunction> hashFunction;
	kj::Array<unsigned char> hashValue;
	
	capnp::ClientHook* registrationKey = nullptr;
	
	kj::List<TransmissionReceiver, &TransmissionReceiver::listLink> receivers;
};

/**
 * Backend implementation of the local data service.
 */
class LocalDataServiceImpl : public kj::Refcounted, public DataService::Server {
public:
	using Nursery = LocalDataService::Nursery;
	using DTContext = DownloadTask<LocalDataRef<capnp::AnyPointer>>::Context;
	
	LocalDataServiceImpl(Library& h);
	Own<LocalDataServiceImpl> addRef();
	
	Promise<LocalDataRef<capnp::AnyPointer>> download(DataRef<capnp::AnyPointer>::Client src, bool recursive, DTContext ctx = DTContext());
	
	LocalDataRef<capnp::AnyPointer> publish(DataRefMetadata::Reader metaData, Array<const byte>&& data, ArrayPtr<Maybe<Own<capnp::ClientHook>>> capTable);
	
	Promise<void> writeArchive(DataRef<capnp::AnyPointer>::Client ref, const kj::File& out);
	
	struct Mappable {
		virtual kj::Array<const kj::byte> mmap(size_t start, size_t size) = 0;
	};
	
	LocalDataRef<capnp::AnyPointer> publishArchive(Mappable& mappable, const capnp::ReaderOptions options);
	
	LocalDataRef<capnp::AnyPointer> publishArchive(const kj::ReadableFile& f, const capnp::ReaderOptions options);
	LocalDataRef<capnp::AnyPointer> publishArchive(const kj::Array<const kj::byte> f, const capnp::ReaderOptions options);
	LocalDataRef<capnp::AnyPointer> publishConstant(const kj::ArrayPtr<const kj::byte> f, const capnp::ReaderOptions options);
	
	Promise<kj::Array<kj::Array<const byte>>> downloadFlat(DataRef<>::Client src);
	LocalDataRef<> publishFlat(kj::Array<kj::Array<const byte>> data);
		
	Promise<void> clone(CloneContext context) override;
	Promise<void> store(StoreContext context) override;
	Promise<void> hash(HashContext context) override;
	
	inline void setLimits(LocalDataService::Limits newLimits);
	
	void setChunkDebugMode();
	
	Promise<Maybe<LocalDataRef<capnp::AnyPointer>>> unwrap(DataRef<capnp::AnyPointer>::Client ref);
	
private:
	capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>> serverSet;
	Library library;
	
	Own<DBCache> dbCache;
	
	LocalDataService::Limits limits;
	MMapTemporary fileBackedMemory;
	
	struct DataRefDownloadProcess;
	
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
	
	using Metadata = DataRefMetadata;
	
	Own<LocalDataRefImpl> addRef();
	
	// Decodes the underlying data as a capnproto message
	template<typename T>
	typename T::Reader get(const capnp::ReaderOptions& options);
	
	// Returns an additional owning ref to the underlying data
	kj::Array<const byte> addRefRaw();
	
	// Returns a reader to the locally stored metadata
	Metadata::Reader localMetadata();
	
	Promise<void> metaAndCapTable(MetaAndCapTableContext) override ;
	Promise<void> rawBytes(RawBytesContext) override ;
	Promise<void> transmit(TransmitContext) override ;
	
	// Reference to the local data store entry holding our data
	StoreEntry entry = nullptr;
	
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

// Helper methods to get schema nodes

template<typename T>
capnp::Type typeFor(typename T::Reader) {
	return capnp::Type::from<T>();
}

template<>
inline capnp::Type typeFor<capnp::DynamicStruct>(capnp::DynamicStruct::Reader r) {
	return r.getSchema();
}

template<>
inline capnp::Type typeFor<capnp::AnyStruct>(capnp::AnyStruct::Reader r) {
	return capnp::schema::Type::AnyPointer::Unconstrained::STRUCT;
}

template<>
inline capnp::Type typeFor<capnp::AnyList>(capnp::AnyList::Reader r) {
	return capnp::schema::Type::AnyPointer::Unconstrained::LIST;
}

template<>
inline capnp::Type typeFor<capnp::AnyPointer>(capnp::AnyPointer::Reader r) {
	return capnp::schema::Type::AnyPointer::Unconstrained::ANY_KIND;
}

template<typename T>
struct References_ { using Type = typename References_<capnp::FromClient<T>>::Type; };

template<typename T>
struct References_<DataRef<T>> { using Type = T; };

template<typename T>
struct References_<LocalDataRef<T>> { using Type = T; };

} // namespace fsc::internal

// === class LocalDataService ===

template<typename Reader, typename T>
LocalDataRef<T> LocalDataService::publish(Reader data) {
	capnp::BuilderCapabilityTable capTable;
	
	Array<const byte> byteData = internal::buildData<T>(data, capTable);
	
	Temporary<DataRefMetadata> metadata;
	metadata.setId(getActiveThread().randomID());
	metadata.setCapTableSize(capTable.getTable().size());
	metadata.setDataSize(byteData.size());
	// dataHash set by impl->publish()
	
	Temporary<capnp::schema::Type> typeBuilder;
	extractType(internal::typeFor<T>(data), typeBuilder);
	
	if(typeBuilder.isData()) {
		metadata.getFormat().setRaw();
	} else {
		metadata.getFormat().initSchema().setAs<capnp::schema::Type>(typeBuilder.asReader());
	}
			
	return impl->publish(
		metadata,
		mv(byteData),
		capTable.getTable()
	).template as<T>();
}

template<typename T>
LocalDataRef<T> LocalDataService::publish(
	typename DataRefMetadata::Reader metaData,
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
		metaData,
		mv(backingArray),
		hooks.finish()
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
Promise<void> LocalDataService::writeArchive(Ref ref, const kj::File& out) {
	return impl -> writeArchive(ref.asGeneric(), out);
}

template<typename T>
LocalDataRef<T> LocalDataService::publishArchive(const kj::ReadableFile& in, const capnp::ReaderOptions options) {
	return impl -> publishArchive(in, options).as<T>();
}

template<typename T>
LocalDataRef<T> LocalDataService::publishArchive(kj::Array<const byte> in, const capnp::ReaderOptions options) {
	return impl -> publishArchive(mv(in), options).as<T>();
}

template<typename T>
LocalDataRef<T> LocalDataService::publishConstant(kj::ArrayPtr<const byte> in) {
	return impl -> publishConstant(mv(in), READ_UNLIMITED).as<T>();
}

template<typename T>
LocalDataRef<T> LocalDataService::publishFlat(kj::Array<kj::Array<const byte>> data) {
	return impl -> publishFlat(mv(data)).as<T>();
}

template<typename C, typename T>
Promise<kj::Array<kj::Array<const byte>>> LocalDataService::downloadFlat(C ref) {
	return impl -> downloadFlat(ref.template castAs<DataRef<>>());
}

template<typename T>
Promise<kj::Array<kj::Array<const byte>>> LocalDataService::downloadFlat(LocalDataRef<T> ref) {
	return impl -> downloadFlat(ref.template as<capnp::AnyPointer>());
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
Array<const byte> LocalDataRef<T>::forkRaw() {
	return backend -> addRefRaw();
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
DataRefMetadata::Format::Reader LocalDataRef<T>::getFormat() {
	return backend -> localMetadata().getFormat();
}

template<typename T>
typename DataRefMetadata::Reader LocalDataRef<T>::getMetadata() {
	return backend -> _metadata.getRoot<DataRefMetadata>();
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

namespace internal {
	template<typename F>
	struct BackoffRunner {
		using ResultPromise = kj::PromiseForResult<F, void>;
		
		kj::Duration current;
		const kj::Duration max;
		const uint64_t growth;
		F f;
		
		BackoffRunner(kj::Duration min, kj::Duration max, uint64_t growth, F func) :
			current(min),
			max(max),
			growth(growth),
			f(mv(func))
		{}
		
		ResultPromise step() {
			return kj::evalLater(f)
			.catch_([this](kj::Exception&& e) -> ResultPromise {
				if(e.getType() == kj::Exception::Type::OVERLOADED) {
					ResultPromise result = getActiveThread().timer().afterDelay(current)
					.then([this]() { return step(); });
					
					current *= growth;
					if(current > max) current = max;
					
					return result;
				}
				
				kj::throwFatalException(mv(e));
			});
		}
	};
}

template<typename F>
kj::PromiseForResult<F, void> withBackoff(kj::Duration min, kj::Duration max, uint64_t growth, F func) {
	auto runner = heapHeld<internal::BackoffRunner<F>>(min, max, growth, mv(func));
	return runner -> step().attach(runner.x());
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
	
	KJ_REQUIRE(!impl.localMetadata().getFormat().isRaw(), "Attempting to read raw DataRef as structured type");
	
	// Return the reader's root at the requested type
	capnp::AnyPointer::Reader root = msgReader.getRoot<capnp::AnyPointer>();
	root = impl.readerTable -> imbue(root);
	
	// Copy root onto the heap and attach objects needed to keep it running
	return root.getAs<T>();
}

// === class DownloadTask ===

namespace internal {

template<typename Result>
DownloadTask<Result>::DownloadTask(DataRef<capnp::AnyPointer>::Client src, Context ctx) :
	src(src), hashFunction(getActiveThread().library() -> defaultHash()), result(nullptr), ctx(ctx)
{
	result = actualTask().fork();
}

template<typename Result>
DownloadTask<Result>::~DownloadTask() {
	if(registrationKey != nullptr) {
		auto& active = ctx.registry -> activeDownloads;
		active.erase(registrationKey);
	}
	
	for(auto& recv : receivers) {
		recv.clear();
	}
}

template<typename Result>
Promise<Result> DownloadTask<Result>::actualTask() {	
	// Check if the result can be directly obtained by unwrapping
	return checkLocalAndRegister().then([this](Maybe<ResultType> result) mutable -> Promise<Result> {
		KJ_IF_MAYBE(pResult, result) {
			return mv(*pResult);
		}
		
		return downloadMetaAndCapTable()		
		.then([this]() mutable {
			// Check if we can use cached data for the download
			return useCached();
		})
		.then([this](Maybe<Result> maybeResult) mutable -> Promise<Result> {
			// If we can use the cache, use that
			// Otherwise, perform download
			KJ_IF_MAYBE(pResult, maybeResult){
				return mv(*pResult);
			}
			
			return downloadData()
			.then([this]() mutable {
				return buildResult();
			});
		});
	});
}

template<typename Result>
Promise<Maybe<Result>> DownloadTask<Result>::checkLocalAndRegister() {
	using capnp::ClientHook;
	
	// Wait for hook to resolve
	return src.whenResolved()
	
	// Check if hook can be unwrapped locally
	.then([this]() mutable { return unwrap(); })
	.then([this](Maybe<ResultType> maybeResult) mutable -> Promise<Maybe<ResultType>> {
		// If unwrap succeeded, return unwrapped
		KJ_IF_MAYBE(pResult, maybeResult) {
			return mv(maybeResult);
		}
		
		// Scan all nested client hooks and check whether
		// they are currently being downloaded (deduplication)
		auto& active = ctx.registry -> activeDownloads;
		
		Own<ClientHook> hook = ClientHook::from(cp(src));
		ClientHook* inner = hook.get();
		
		while(true) {				
			KJ_IF_MAYBE(pNested, inner -> getResolved()) {
				inner = &(*pNested);
			} else {
				break;
			}
		}
			
		auto it = active.find(inner);
		if(it != active.end()) {
			return it -> second -> output()
			.then([](ResultType result) -> Maybe<ResultType> { return result; });
		} else {
			registrationKey = inner;
			active.insert(std::make_pair(inner, this));
		}
			
		return Maybe<ResultType>(nullptr);
	});
}

template<typename Result>
Promise<void> DownloadTask<Result>::downloadMetaAndCapTable() {	
	return src.metaAndCapTableRequest().send()
	.then([this](auto response) mutable {
		metadata = response.getMetadata();
		
		auto capTable = response.getTable();
		auto builder = kj::heapArrayBuilder<capnp::Capability::Client>(capTable.size());
		for(auto ref : capTable) {
			auto adjusted = adjustRef(ref);
			
			auto clientPromise = adjusted.whenResolved()
			.then([adjusted]() mutable { return adjusted;  })
			.catch_([ref](kj::Exception&& e) mutable {				
				if(e.getType() == kj::Exception::Type::UNIMPLEMENTED)
					return ref;
				throw e;
			});
			
			builder.add(mv(clientPromise));
		}
		
		this -> capTable = builder.finish();
	});
}

template<typename Result>
Promise<void> DownloadTask<Result>::downloadData() {
	auto downloadRequest = src.transmitRequest();
	
	downloadRequest.setStart(0);
	downloadRequest.setEnd(metadata.getDataSize());
	downloadRequest.setReceiver(kj::heap<TransmissionReceiver>(*this));
	
	return downloadRequest.send().ignoreResult();
}

// ===== class DownloadTask::TransmissionReceiver =====

template<typename Result>
DownloadTask<Result>::TransmissionReceiver::TransmissionReceiver(DownloadTask<Result>& parent) :
	parent(parent)
{
	parent.receivers.add(*this);
}

template<typename Result>
DownloadTask<Result>::TransmissionReceiver::~TransmissionReceiver() {
	clear();
}

template<typename Result>
Promise<void> DownloadTask<Result>::TransmissionReceiver::begin(BeginContext ctx) {
	KJ_IF_MAYBE(pParent, parent) {
		auto ka = pParent -> addRef();
		return pParent -> beginDownload().attach(mv(ka));
	}
	KJ_FAIL_REQUIRE("Download task cancelled");
}

template<typename Result>
Promise<void> DownloadTask<Result>::TransmissionReceiver::receive(ReceiveContext ctx) {
	KJ_IF_MAYBE(pParent, parent) {
		auto ka = pParent -> addRef();
		auto data = ctx.getParams().getData();
		pParent -> hashFunction -> update(data.begin(), data.size());
		return pParent -> receiveData(data).attach(mv(ka));
	}
	KJ_FAIL_REQUIRE("Download task cancelled");
}

template<typename Result>
Promise<void> DownloadTask<Result>::TransmissionReceiver::done(DoneContext ctx) {
	KJ_IF_MAYBE(pParent, parent) {
		auto ka = pParent -> addRef();
		pParent -> hashValue = kj::heapArray<unsigned char>(pParent -> hashFunction -> output_length());
		pParent -> hashFunction -> final(pParent -> hashValue.begin());
		
		pParent -> metadata.setDataHash(pParent -> hashValue.asBytes());
		
		return pParent -> finishDownload().attach(mv(ka));
	}
	KJ_FAIL_REQUIRE("Download task cancelled");
}

template<typename Result>
void DownloadTask<Result>::TransmissionReceiver::clear() {
	KJ_IF_MAYBE(pParent, parent) {
		pParent -> receivers.remove(*this);
	}
	parent = nullptr;
}

}

}