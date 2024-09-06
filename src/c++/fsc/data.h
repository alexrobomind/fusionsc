#pragma once

#include <fsc/data.capnp.h>

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <kj/async.h>
#include <limits>
//#include <kj/filesystem.h>

#include "common.h"
#include "local.h"

/**
 * \defgroup network Distributed data and networking
 *
 * The FusionSC service model is based around using Cap'n'proto capabilities. Large data objects are represented
 * using a generic service interface called \ref fsc::DataRef "DataRef". Objects of this type are passed by reference - no data are
 * copied at this stage.
 *
 * When backed by local memory, DataRefs are represented as a subclass of DataRef::Client called \ref fsc::LocalDataRef "LocalDataRef".
 * This class permits access to the stored Cap'n'proto object, underlying raw storage, and referenced DataRef
 * and/or LocalDataRef objects.
 *
 * Converting a DataRef into a LocalDataRef (possibly) requires consulting a local data table for de-duplication,
 * as well as perhaps downloading and re-hashing the requested data. This process is managed by the
 * \ref fsc::LocalDataService "LocalDataService" class.
 *
 * \code
 * using namespace fsc;
 * ...
 * // Obtain a wait scope so we can use synchronous API
 * kj::WaitScope& ws = ...;
 * DataRef<Float64Tensor>::Client ref = ...;
 *
 * // Download the reference to local memory
 * LocalDataService& service = getActiveThread().dataService();
 * LocalDataRef<Float64Tensor> localRef = service.download(ref).wait(ws);
 *
 * // Access data
 * Float64Tensor::Reader localData = localRef.get();
 * \endcode
 * For exchanging tensor information, the data module defines the following Cap'n'proto struct types:
 *
 * \snippet data.capnp tensors
 */

namespace kj {
	class File;
	class ReadableFile;
}

namespace fsc {

// Internal forward declarations

namespace internal {
	class LocalDataRefImplV2;
	class LocalDataServiceImpl;
	
	template<typename T>
	struct References_;
	
	// Specialized in data-inl.h
	template<typename T>
	struct TensorFor_ {};
}

// API forward declarations

template<typename T> class LocalDataRef;
class LocalDataService;

//! Use this to figure out what datatype a DataRef points to.
template<typename T>
using References = typename internal::References_<T>::Type;

// ============================================ API =============================================

#ifdef DOXYGEN

//! Cap'n'proto interface for data objects.
/** 
 * \ingroup network
 * \tparam T Type of the root message stored in the data ref.
 *
 * The DataRef template is a special capability recognized all throughout the FSC library.
 * It represents a link to a data storage location (local or remote), associated with a
 * unique ID, which can be downloaded to local storage and accessed there. Locally downloaded
 * data are represented by the LocalDataRef class, which subclasses DataRef::Client.
 *
 * The stored data is expected to be in one of the two following formats:
 * - If T is any type except capnp::Data, it must correspond to a Cap'n'proto struct type. In
 *   this case, the DataRef holds a message with the corresponding root type, and
 *   a capability table that tracks any remote objects in use by this message
 *  (including other DataRef instances).
 * - If T is DataRef::Data, then the corresponding message can either be a raw binary or a
     Cap'n'proto message with a capnp::Data object as its root (note that the library will
	 default to raw storage because of the laxer size constraints)
 *
 * Once obtained, DataRefs can be freely passed around as part of RPC calls or data published
 * in other DataRef::Client instances. The fsc runtime will do all it possibly can to protect the integrity
 * of a DataRef. In the absence of hardware failure, data referenced via DataRef objects
 * can only go out of use once all referencing DataRef objects do so as well.
 *
 * DataRefs only represent a link to locally or remotely stored data. To access the underlying
 * data, they must be converted into LocalDataRef instances using LocalDataService::download() methods.
 *
 * \capnpinterface
 * \snippet data.capnp DataRef
 *
 */
template <typename T = ::capnp::AnyPointer>
struct DataRef {
	//! Actual interface object class
	struct Client : public virtual ::capnp::Capability::Client {
		template <typename T2 = ::capnp::AnyPointer>
		typename DataRef<T2>::Client asGeneric() {
			return castAs<DataRef<T2>>();
		}
	};
};

//! Remote interface to data service
/**
 * \ingroup network
 *
 * This interface serves as an access point to remotely download 
 *
 * \capnpinterface
 * \snippet data.capnp DataService
 */
struct DataService {
};

#endif

constexpr capnp::ReaderOptions READ_UNLIMITED { std::numeric_limits<uint64_t>::max(), std::numeric_limits<int>::max() };

//! Checks if the object in question represents a DataRef object
Promise<bool> isDataRef(capnp::Capability::Client);

//! Publishes and downloads data into LocalDataRef instances.
/**
 * \ingroup network
 * Main entry point for handling local and remote data references. Can be used to both create
 * remotely-downloadable data references with its 'publish' methods and download (as in, create
 * local copies of) remote references with its 'download' methods.
 */
class LocalDataService {
public:
	using Nursery = kj::Vector<kj::Own<void>>;
	
	//! \name Download methods
	///@{
	
	//! Downloads remote DataRef::Client into LocalDataRef
	/**
	 * Downloads the data contained in the remote reference into the local backing
	 * store and links the remote capabilities into a local capability table.
	 *
	 * \param src The DataRef<T>::Client to download from (can also be a LocalDataRef<T>)
	 * \param recursive Whether to recursively download all referenced DataRef instances
	 *        into the local data store as well. If true, all contained DataRef objects will
	 *        point into local storage after the returned promise resolved, and are therefore
	 *        guaranteed to instantly resolve in future download attempts.
	 *
	 * \returns Promise to a local data ref instance which extends the interface by DataRef
	 * with direct access to the stored data.
	 */
	template<typename Reference, typename T = References<Reference>>
	Promise<LocalDataRef<T>> download(Reference src, bool recursive = false);
	
	//! Downloads remote DataRef::Client into LocalDataRef if it is not null
	/**
	 * Downloads the data contained in the remote reference into the local backing
	 * store and links the remote capabilities into a local capability table.
	 *
	 * \param src The DataRef<T>::Client to download from (can also be a LocalDataRef<T>)
	 * \param recursive Whether to recursively download all referenced DataRef instances
	 *        into the local data store as well. If true, all contained DataRef objects will
	 *        point into local storage after the returned promise resolved, and are therefore
	 *        guaranteed to instantly resolve in future download attempts.
	 *
	 * \returns Promise to a local data ref instance which extends the interface by DataRef
	 * with direct access to the stored data. The contained Maybe will evaluate to nullptr
	 * if the given DataRef is unset.
	 */
	template<typename Reference, typename T = References<Reference>>
	Promise<Maybe<LocalDataRef<T>>> downloadIfNotNull(Reference src, bool recursive = false);
	
	///@}
	
	//! \name Publication methods
	///@{
	
	//! Publishes binary data and capability table
	/**
	 * Creates a local data reference directly from a backing array and a capability table.
	 *
	 * The interpretation of the backing array depends on the seleced data type. If the
	 * data type is capnp::Data, the array is interpreted as the raw data intended to
	 * be referenced. This is e.g. intended to be used for raw memory-mapped files.
	 * Currently, any other type leads to the backing array being interpreted as containing
	 * a CapNProto message (including its segment table) with a root of the specified data
	 * type (can also be capnp::AnyPointer, capnp::AnyList or capnp::AnyStruct or similar).
	 *
	 * \param id The global ID to publish this data ref under.
	 * \param backingArray The contents of the binary buffer. The LocalDataRef object will
	 *        take ownership of this buffer.
	 * \param capTable A list of capabilities (usually the capability table of the passed
	 *        Cap'n'Proto message) to be stored alongside the binary data.
	 *
	 * \returns A LocalDataRef object which can be passed into all targets expecting abort
	 *          DataRef::Client.
	 */
	template<typename T = capnp::Data>
	LocalDataRef<T> publish(typename DataRefMetadata::Reader metaData, Array<const byte> backingArray, ArrayPtr<capnp::Capability::Client> capTable = nullptr);
	
	//! Publishes Cap'n'proto message
	/**
	 * Creates a local data reference by copying the contents of a capnproto reader.
	 * If the reader is of type capnp::Data, the byte array it points to will be copied
	 * verbatim into the backing buffer.
	 * Currently, for any other type this method will create a message containing a deepcopy
	 * copy of the data referenced by this reader and store it into the backing array (including
	 * the message's segment table).
	 * Capabilities contained in the reader's message will be added into the capability table
	 * hosted by the DataRef.
	 */
	template<typename Reader, typename T = capnp::FromAny<Reader>>
	LocalDataRef<T> publish(Reader reader);
	
	//! Publish the contents of an array (copies and hashes the array)
	LocalDataRef<capnp::Data> publish(kj::ArrayPtr<const byte> bytes);
	
	//! Take ownership of array and publish it, potentially with precomputed hash
	LocalDataRef<capnp::Data> publish(kj::Array<const byte> bytes, kj::ArrayPtr<const kj::byte> hash = nullptr);
	
	///@}
	
	//! \name Archiving methods
	///@{
	
	//! Write DataRef to an archive file
	/**
	 * Downloads the target data and all its transitive dependencies and writes them
	 * into an archive file. This file can then be shared with other customers to provide
	 * them a deep copy of the stored data.
	 */
	template<typename Ref, typename T = References<Ref>>
	Promise<void> writeArchive(Ref reference, const kj::File& out) KJ_WARN_UNUSED_RESULT;	
	
	//! Publish contents of archive file as LocalDataRef
	/**
	 * Reads an archive file and publishes all data contained within. Returns a LocalDataRef
	 * to the root used when writing the archive.
	 */
	template<typename T>
	LocalDataRef<T> publishArchive(const kj::ReadableFile& in, const capnp::ReaderOptions readerOpts = READ_UNLIMITED);
	
	template<typename T>
	LocalDataRef<T> publishArchive(kj::Array<const byte> in, const capnp::ReaderOptions readerOpts = READ_UNLIMITED);
	
	template<typename T>
	LocalDataRef<T> publishConstant(kj::ArrayPtr<const byte> in);
	
	//! Publish the raw contents of a file as a data ref via mmap or copy
	LocalDataRef<capnp::Data> publishFile(const kj::ReadableFile& in, kj::ArrayPtr<const kj::byte> fileHash = nullptr, bool copy = false);
	
	//! Shorthand for publishing without hash
	LocalDataRef<capnp::Data> publishFile(const kj::ReadableFile& in, bool copy = false);
	
	///@}
	
	//! \name Flat representation
	///@{
	
	template<typename Client>
	Promise<kj::Array<kj::Array<const byte>>> downloadFlat(Client src);
	
	template<typename T>
	Promise<kj::Array<kj::Array<const byte>>> downloadFlat(LocalDataRef<T> src);
	
	template<typename T>
	LocalDataRef<T> publishFlat(kj::Array<kj::Array<const byte>> data);
	
	///@}
	
	//! \name Raw data files
	///@{
	
	Promise<void> downloadIntoFile(DataRef<capnp::Data>::Client, Own<const kj::File>&& out);
	
	///@}
	
	//! \name Limit configuration
	///@{
		
	struct Limits {
		uint64_t maxRAMObjectSize = 100000000; // Store up to 100MB in Ram
		Maybe<uint64_t> ramRemaining = nullptr; // Sets the remaining RAM budget. After this amount of RAM is requested, switch to file-backed allocation
	};
	
	void setLimits(Limits limits);	
	
	///@}
	
	//! Reduces chunk size to 1kB and throws error if chunks can't be mapped
	void setChunkDebugMode();
	
	operator DataService::Client();
	
	/**
	 * Constructs a new data service instance using the shared backing store.
	 */
	LocalDataService(const LibraryHandle& hdl);

	// Non-const copy constructor
	LocalDataService(LocalDataService& other);
	
	// Move constructor
	LocalDataService(LocalDataService&& other) = default;

	// Copy assignment operator
	LocalDataService& operator=(LocalDataService& other);

	// Copy assignment operator
	LocalDataService& operator=(LocalDataService&& other) = default;
	
	LocalDataService() = delete;
		
private:
	Own<internal::LocalDataServiceImpl> impl;
	
	LocalDataService(internal::LocalDataServiceImpl& impl);
	
	template<typename T>
	friend class LocalDataRef;
};

//! Local version of DataRef::Client.
/**
 * \ingroup network
 * Data reference backed by local storage. In addition to the remote access functionality
 * provided by the interface in capnp::DataRef, this class provides direct access to
 * locally stored data.
 * This class uses non-atomic reference counting for performance, so it can not be
 * shared across threads. To share this to other threads, pass the DataRef::Client
 * capability it inherits from via RPC and use that thread's DataService to download
 * it into a local reference. If this ref and the other DataServce share the same
 * data store, the underlying data will not be copied, but shared between the references.
 */
template<typename T = capnp::AnyPointer>
class LocalDataRef : public DataRef<T>::Client {
public:
	using typename DataRef<T>::Client::Calls;
	
	/**
	 * Provides direct access to the raw underlying byte array associated
	 * with this data reference.
	 */
	ArrayPtr<const byte> getRaw();
	
	/**
	 * Same as getRaw(), but provides an owning reference (that can be passed across threads)
	 */
	Array<const byte> forkRaw();
	
	/**
	 * Provides a structured view of the underlying data. If T is capnp::Data,
	 * the returned reader will be identical to getRaw(). Otherwise, this will
	 * interpret the backing array as a CapNProto message with the given type
	 * at its root. Note that if T is not capnp::Data, and the backing array
	 * can not be interpreted as a CapNProto message, this method will fail.
	 */
	typename T::Reader get(const capnp::ReaderOptions& options = READ_UNLIMITED);
	
	/**
	 * Provides a new data reference sharing the underling buffer and
	 * capabilities, but having a different interpretation data type.
	 */
	template<typename T2 = capnp::AnyPointer>
	class LocalDataRef<T2> as();

	ArrayPtr<const byte> getID();
	ArrayPtr<capnp::Capability::Client> getCapTable();
	DataRefMetadata::Format::Reader getFormat();
	
	typename DataRefMetadata::Reader getMetadata();

	// Non-const copy constructor
	LocalDataRef(LocalDataRef<T>& other);
	
	// Move constructor
	LocalDataRef(LocalDataRef<T>&& other);

	// Copy assignment operator
	LocalDataRef<T>& operator=(LocalDataRef<T>&  other);
	
	// Move assignment operator
	LocalDataRef<T>& operator=(LocalDataRef<T>&& other);

	LocalDataRef() = delete;
	
	LocalDataRef(DataRef<capnp::AnyPointer>::Client capView, Own<internal::LocalDataRefImplV2> backend);

	template<typename T2>	
	LocalDataRef(LocalDataRef<T2>& other);
	
	Own<internal::LocalDataRefImplV2> backend;
	
	friend class internal::LocalDataServiceImpl;
	
	template<typename T2>
	friend class LocalDataRef;
};

//! Combined MessageBuilder and Struct::Builder
/**
 * \tparam T Data type of the contained message.
 * \ingroup network
 *
 * This class holds a locally owned MessageBuilder and specifies
 * the type of the message's root. It directly derives from T::Builder,
 * so the root struct / list can be directly accessed.
 */
template<typename T>
struct Temporary : public T::Builder {
	using Builds = T;
	
	template<typename... Params>
	Temporary(Params... params) :
		T::Builder(nullptr),
		holder(kj::heap<capnp::MallocMessageBuilder>())
	{
		T::Builder::operator=(holder->initRoot<T>(params...));
	}
	
	Temporary(typename T::Reader reader) :
		T::Builder(nullptr),
		holder(kj::heap<capnp::MallocMessageBuilder>())
	{
		holder->setRoot(cp(reader));
		T::Builder::operator=(holder->getRoot<T>());
	}
	
	Temporary(typename T::Builder builder) :
		Temporary(builder.asReader())
	{}
	
	Temporary(Own<capnp::MessageBuilder> holder) :
		T::Builder(holder->getRoot<T>()),
		holder(mv(holder))
	{}
	
	Temporary(Temporary<T>&&) = default;
	Temporary<T>& operator=(Temporary<T>&& other) = default;
	
	Temporary<T>& operator=(typename T::Reader other) {
		return (*this) = Temporary<T>(other);
	}
	
	Temporary<T>& operator=(std::nullptr_t) {
		holder = kj::heap<capnp::MallocMessageBuilder>();
		T::Builder::operator=(holder->getRoot<T>());
		
		return *this;
	}
	
	typename T::Builder asBuilder() { return *this; }
	
	operator capnp::MessageBuilder&() { return *holder; }
	
	Own<capnp::MessageBuilder> holder;
};

template<typename T>
struct _IsTemporary { constexpr static bool val = false; };

template<typename T>
struct _IsTemporary<fsc::Temporary<T>> { constexpr static bool val = true; };

template<typename T>
constexpr bool isTemporary() { return _IsTemporary<T>::val; }

template<typename T, typename Cap = capnp::FromClient<T>, typename... Attachments>
typename Cap::Client attachToClient(T src, Attachments&&... attachments);

bool hasMaximumOrdinal(capnp::DynamicStruct::Reader in, unsigned int maxOrdinal);

template<typename T, typename Cap = capnp::FromClient<T>, typename... Attachments>
typename Cap::Client attach(T src, Attachments&&... attachments);

//! Helper function for calculating linear index based on shape info
size_t linearIndex(const capnp::List<uint64_t>::Reader& shape, const ArrayPtr<size_t> index);

//! Struct version checker
/**
 * When passing structured data between functions, Cap'n'proto will silently hide all fields which
 * are not understood by the current version of the protocol. Likewise, if new fields are added,
 * methods that do not yet understand them will likely not check for their presence. In many cases,
 * this is useful behavior.
 *
 * However, for scientific codes, when data are requested, it is extremely important that a given
 * request is understood completely. This method aims to enable protocol evolution with this contraint,
 * by checking whether a function, that can only interpret fields up to the given ordinal number in the
 * given struct, will be able to fully understand the passed data. It does so by not only inspecting
 * the ordinal numbers of set fields, but also by inspecting the wire representation of the struct, to
 * ensure that there are no unknown fields set by a potentially newer version of the protocol.
 *
 * \param in The data to be checked for consistency.
 * \param maxOrdinal An upper bound on the ordinal number of fields. Any field with an ordinal number
 *        exceeding this parameter (or unknown to this client) may only be set to its default value,
 *        otherwise this method will return false.
 *
 * \returns true if the given struct meets the max ordinal requirements, otherwise false.
 */
template <typename T, typename = kj::EnableIf<capnp::kind<capnp::FromReader<T>>() == capnp::Kind::STRUCT>>
bool hasMaximumOrdinal(T in, unsigned int maxOrdinal) {
	return hasMaximumOrdinal(capnp::DynamicStruct::Reader(in), maxOrdinal);
}

Promise<void> removeDatarefs(capnp::AnyPointer::Reader in, capnp::AnyPointer::Builder out);
Promise<void> removeDatarefs(capnp::AnyStruct::Reader in, capnp::AnyStruct::Builder out);

template<typename F>
kj::PromiseForResult<F, void> withBackoff(kj::Duration min, kj::Duration max, uint64_t growth, F func);

//! Creates ID from canonical representation of reader
template<typename T>
ID ID::fromReader(T t) {
	return ID(wordsToBytes(capnp::canonicalize(t)));
}

//! Creates ID from canonical representation of reader, replaces contained DataRef::Client s with their IDs.
template<typename T>
Promise<ID> ID::fromReaderWithRefs(T t) {
	Temporary<capnp::FromAny<T>> tmp;
	
	auto stripped = removeDatarefs(capnp::toAny(t), capnp::toAny(tmp));
	
	return stripped.then([tmp = mv(tmp)]() mutable {
		return ID::fromReader(tmp.asReader());
	});
}

//! Provides an overlay over the input data ref that changes the references
DataRef<capnp::AnyPointer>::Client overrideRefs(DataRef<capnp::AnyPointer>::Client, kj::Array<capnp::Capability::Client>);

}

#include "data-inl.h"
