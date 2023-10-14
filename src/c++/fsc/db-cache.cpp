#include "db-cache.h"
#include "blob-store.h"

namespace fsc {

namespace {
	
template<typename T>
auto withCacheBackoff(T func) {
	return withBackoff(10 * kj::MILLISECONDS, 5 * kj::MINUTES, 2, mv(func));
}

struct DBCacheImpl : public DBCache, kj::Refcounted {
	Own<DBCache> addRef() override;
	Promise<DataRef<capnp::AnyPointer>::Client> cache(DataRef<capnp::AnyPointer>::Client target) override;
	
	capnp::CapabilityServerSet<DataRef<capnp::AnyPointer>> serverSet;
	Own<BlobStore> store;
	
	DBCacheImpl(BlobStore& store);
};

struct CachedRef : public DataRef<capnp::AnyPointer>::Server {
	kj::UnwindDetector ud;
	
	Temporary<DataRefMetadata> _metadata;
	kj::Array<capnp::Capability::Client> refs;
	Own<Blob> blob;
	Own<DBCacheImpl> parent;
	
	CachedRef(Temporary<DataRefMetadata>&& metadata, kj::Array<capnp::Capability::Client> refs, Own<Blob> blob, DBCacheImpl& parent);
	~CachedRef();
	
	Promise<void> metaAndCapTable(MetaAndCapTableContext ctx) override;
	Promise<void> rawBytes(RawBytesContext ctx) override;
	Promise<void> transmit(TransmitContext ctx) override;
};

struct DownloadProcess : public internal::DownloadTask<DataRef<capnp::AnyPointer>::Client> {
	Own<DBCacheImpl> parent;
	
	kj::UnwindDetector ud;
	
	kj::OneOf<Own<BlobBuilder>, Own<Blob>, decltype(nullptr)> blobOrBuilder = nullptr;
	
	DownloadProcess(DBCacheImpl& parent, DataRef<capnp::AnyPointer>::Client src);
	~DownloadProcess();
	
	Promise<Maybe<ResultType>> unwrap() override;
	capnp::Capability::Client adjustRef(capnp::Capability::Client ref) override;
	Promise<Maybe<ResultType>> useCached() override;
	
	Promise<void> beginDownload() override;
	Promise<void> receiveData(kj::ArrayPtr<const byte> data) override;
	Promise<void> finishDownload() override;
	
	Promise<ResultType> buildResult() override;
};

struct TransmissionProcess {
	constexpr static inline size_t CHUNK_SIZE = 1024 * 1024;
	
	Own<kj::InputStream> reader;
	
	DataRef<capnp::AnyPointer>::Receiver::Client receiver;
	size_t start;
	size_t end;
	
	Array<byte> buffer;
	
	TransmissionProcess(Own<kj::InputStream>&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end);
	
	Promise<void> run();
	Promise<void> transmit(size_t chunkStart);
};

// class TransmissionProcess
	
TransmissionProcess::TransmissionProcess(Own<kj::InputStream>&& reader, DataRef<capnp::AnyPointer>::Receiver::Client receiver, size_t start, size_t end) :
	reader(mv(reader)),
	receiver(mv(receiver)),
	buffer(kj::heapArray<byte>(CHUNK_SIZE)),
	start(start), end(end)
{
	KJ_REQUIRE(end >= start);
}

Promise<void> TransmissionProcess::run() {
	reader -> skip(start);
	
	auto request = receiver.beginRequest();
	request.setNumBytes(end - start);
	return request.send().ignoreResult().then([this]() { return transmit(start); });
}

Promise<void> TransmissionProcess::transmit(size_t chunkStart) {			
	// Check if we are done transmitting
	if(chunkStart >= end)
		return receiver.doneRequest().send().ignoreResult();
	
	auto slice = chunkStart + CHUNK_SIZE <= end ? buffer.asPtr() : buffer.slice(0, end - chunkStart);
	reader -> read(slice.begin(), slice.size());
	
	// Do a transmission
	auto request = receiver.receiveRequest();
	
	if(slice.size() % 8 == 0) {
		// Note: This is safe because we keep this object alive until the transmission
		// succeeds or fails
		auto orphanage = capnp::Orphanage::getForMessageContaining((DataRef<capnp::AnyPointer>::Receiver::ReceiveParams::Builder) request);
		auto externalData = orphanage.referenceExternalData(slice);
		request.adoptData(mv(externalData));
	} else {
		request.setData(slice);
	}
	
	return request.send().then([this, chunkEnd = chunkStart + slice.size()]() { return transmit(chunkEnd); });
}

// class DBCacheImpl

DBCacheImpl::DBCacheImpl(BlobStore& store) :
	store(store.addRef())
{}

// class CachedRef
	
//! Create a cached reference by stealing metadata, refs AND blob (so no incRef inside, but a decRef on destruction)
CachedRef::CachedRef(Temporary<DataRefMetadata>&& metadata, kj::Array<capnp::Capability::Client> refs, Own<Blob> blob, DBCacheImpl& parent) :
	_metadata(mv(metadata)),
	refs(mv(refs)),
	blob(mv(blob)),
	parent(kj::addRef(parent))
{
	// No incRef, since the constructor steals the blob's refcount
}

CachedRef::~CachedRef() {
	ud.catchExceptionsIfUnwinding([this]() mutable {
		blob -> decRef();
	});
}

Promise<void> CachedRef::metaAndCapTable(MetaAndCapTableContext ctx) {
	// The metadata table is only ready once the hash is verified
	ctx.initResults().setMetadata(_metadata);
	
	auto tableOut = ctx.getResults().initTable(refs.size());
	for(auto i : kj::indices(refs))
		tableOut.set(i, refs[i]);
	
	return READY_NOW;
}

Promise<void> CachedRef::rawBytes(RawBytesContext ctx) {
	return withCacheBackoff([this, ctx]() mutable {			
		const uint64_t start = ctx.getParams().getStart();
		const uint64_t end = ctx.getParams().getEnd();
		KJ_REQUIRE(end >= start);
		KJ_REQUIRE(end < _metadata.getDataSize());
		
		auto buffer = kj::heapArray<byte>(8 * 1024 * 1024);
		auto reader = blob -> open();
		reader -> skip(start);
		
		auto data = ctx.getResults().initData(end - start);
		reader -> read(data.begin(), data.size());
	});
}

Promise<void> CachedRef::transmit(TransmitContext ctx) {
	return withCacheBackoff([this, ctx]() mutable {
		auto params = ctx.getParams();
		auto reader = blob -> open();
		auto transProc = heapHeld<TransmissionProcess>(mv(reader), params.getReceiver(), params.getStart(), params.getEnd());
		
		auto transmission = kj::evalNow([=]() mutable { return transProc -> run(); });
		return transmission.attach(transProc.x());
	})
	.attach(thisCap());
}

// class DownloadProcess
	
DownloadProcess::DownloadProcess(DBCacheImpl& parent, DataRef<capnp::AnyPointer>::Client src) :
	DownloadTask(src, Context()),
	parent(kj::addRef(parent)),
	blobOrBuilder(nullptr)
{}

DownloadProcess::~DownloadProcess() {
	ud.catchExceptionsIfUnwinding([this]() {
		if(blobOrBuilder.is<Own<Blob>>()) {
			blobOrBuilder.get<Own<Blob>>() -> decRef();
		} else if(blobOrBuilder.is<Own<BlobBuilder>>()) {
			blobOrBuilder.get<Own<BlobBuilder>>() -> getBlobUnderConstruction() -> decRef();
		}
	});
}
	
//! Check whether "src" can be directly unwrapped
Promise<Maybe<DownloadProcess::ResultType>> DownloadProcess::unwrap() {
	return parent -> serverSet.getLocalServer(src)
	.then([this](Maybe<DataRef<capnp::AnyPointer>::Server> maybeResult) -> Maybe<ResultType> {
		KJ_IF_MAYBE(pResult, maybeResult) {
			auto backend = static_cast<CachedRef*>(pResult);
			
			// Ensure that the backend comes from the same store
			// TODO: I should just have a non-static server set here ...
			if(backend -> parent.get() != parent.get())
				return nullptr;
			
			return src;
		}
		
		return nullptr;
	});
}

//! Adjust refs e.g. by performing additional downloads. If the resulting client is broken with an exception of type "unimplemented", the original ref is used instead.
capnp::Capability::Client DownloadProcess::adjustRef(capnp::Capability::Client ref) {
	return mv(ref); //DBCache doesn't do recursive downloads
}

Promise<Maybe<DownloadProcess::ResultType>> DownloadProcess::useCached() {
	return withCacheBackoff([this]() mutable -> Maybe<ResultType> {
		// Check if blob is already present in store
		auto hash = metadata.getDataHash();
		KJ_IF_MAYBE(pBlob, parent -> store -> find(hash)) {
			(**pBlob).incRef();
			
			auto server = kj::heap<CachedRef>(mv(metadata), mv(capTable), mv(*pBlob), *parent);
			auto client = parent -> serverSet.add(mv(server));
			return client;
		}
		
		return nullptr;
	});
}

Promise<void> DownloadProcess::beginDownload() {
	return withCacheBackoff([this]() mutable {
		constexpr size_t CHUNK_SIZE = 100 * 1024;
		blobOrBuilder = parent -> store -> create(CHUNK_SIZE);
	});
}

Promise<void> DownloadProcess::receiveData(kj::ArrayPtr<const byte> data) {
	return withCacheBackoff([this, data]() mutable {		
		blobOrBuilder.get<Own<BlobBuilder>>() -> write(data.begin(), data.size());
	});
}

Promise<void> DownloadProcess::finishDownload() {
	return withCacheBackoff([this]() mutable {		
		blobOrBuilder = blobOrBuilder.get<Own<BlobBuilder>>() -> finish();
	});
}

Promise<DownloadProcess::ResultType> DownloadProcess::buildResult() {
	return withCacheBackoff([this]() mutable {
		auto server = kj::heap<CachedRef>(
			mv(metadata),
			mv(capTable),
			mv(blobOrBuilder.get<Own<Blob>>()),
			*parent
		);
		
		// The CachedRef steals the blob, so we need to set blob to
		// nullptr to avoid the destructor double-freeing
		blobOrBuilder = nullptr;
		return parent -> serverSet.add(mv(server));
	});
}

}

Own<DBCache> createCache(BlobStore& store) {
	return kj::refcounted<DBCacheImpl>(store);
}

}