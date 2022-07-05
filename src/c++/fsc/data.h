#pragma once

#include <fsc/data.capnp.h>

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <kj/async.h>
//#include <kj/filesystem.h>

#include "common.h"
#include "local.h"

/**
 * \defgroup network Distributed data and networking
 * 
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
	class LocalDataRefImpl;
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

/** 
 * \ingroup network
 * \tparam T Type of the root message stored in the data ref.
 *
 * The DataRef template is a special capability recognized all throughout the FSC library.
 * It represents a link to a data storage location (local or remote), associated with abort
 * unique ID, which can be downloaded to local storage and accessed there. Locally downloaded
 * data are represented by the LocalDataRef class, which subclasses DataRef::Client.
 *
 * The stored data is expected to be in one of the two following formats:
 * - If T is capnp::Data, then the dataref stores a raw binary data array
 * - If T is any other type, it must correspond to a Cap'n'proto struct type. In
 *   this case, the DataRef holds a message with the corresponding root type, and
 *   a capability table that tracks any remote objects in use by this message
 *  (including other DataRef instances).
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
 *
 */
template <typename T = ::capnp::AnyPointer>
struct DataRef {
	class Client : public virtual ::capnp::Capability::Client {
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
 * \snippet data.capnp DataService
 * \capnpinterface
 */
struct DataService {
};

#endif

//! Publishes and downloads data into LocalDataRef instances.
/**
 * \ingroup network
 * Main entry point for handling local and remote data references. Can be used to both create
 * remotely-downloadable data references with its 'publish' methods and download (as in, create
 * local copies of) remote references with its 'download' methods.
 */
class LocalDataService : public DataService::Client {
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
	Promise<LocalDataRef<T>> download(Reference src, bool recursive = true);
	
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
	LocalDataRef<T> publish(ArrayPtr<const byte> id, Array<const byte> backingArray, ArrayPtr<Maybe<Own<capnp::Capability::Client>>> capTable = kj::heapArrayBuilder<Maybe<Own<capnp::Capability::Client>>>(0).finish());
	
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
	LocalDataRef<T> publish(ArrayPtr<const byte> id, Reader reader);
	
	//! Publishes method with ID derived from first argument
	/**
	 * Creates a local data reference by copying the contents of the second argument.
	 * The ID of the reference is derived from the first argument. This process requires
	 * inspecting the ID of DataRef instances contained in the first argument, which might
	 * involve remote calls. Therefore, only a promise is returned.
	 *
	 * \warning The second argument must be a valid until the returned promise resolves, as it
	 * is only copied for publication after the ID is computed.
	 *
	 * \param dataForID Input from which the ID of this object will be derived. While this
	 *        reader may contain DataRef s, no other type of Capability may be stored in
	 *        its message. Currently, the following information is hashed to obtain the stored ID:
	 *         - The Cap'n'Proto type ID of the given reader
	 *         - The canonical representation of the reader, after replacing all
	 *           contained DataRef instances with their IDs.
	 * \param data The data to be stored under the ID computed from the first argument.
	 
	 * \param hashFunction The name of the hash function to be used for ID computation. Must be
	 *        a valid hash function name for the Botan library.
	 *
	 * \returns A promise to a LocalDataRef, which will resolve once the information to compute
	 *          the ID has been obtained.
	 */
	template<typename Reader, typename IDReader, typename T = capnp::FromAny<Reader>, typename T2 = capnp::FromAny<IDReader>>
	Promise<LocalDataRef<T>> publish(IDReader dataForID, Reader data, kj::StringPtr hashFunction = "SHA-256"_kj) KJ_WARN_UNUSED_RESULT;
	
	
	//! Shorthand for publish(data, data, hashFunction)
	/**
	 * Creates a local data reference with an ID derived from the contents.
	 * WARNING: The first argument must be kept alive until the returned promise resolves.
	 */
	template<typename Reader, typename T = capnp::FromAny<Reader>>
	KJ_WARN_UNUSED_RESULT Promise<LocalDataRef<T>> publish(Reader data, kj::StringPtr hashFunction = "SHA-256"_kj) {
		return publish(data, data, hashFunction);
	}
	
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
	LocalDataRef<T> publishArchive(const kj::ReadableFile& in);
	
	//! Write DataRef to an Archive::Builder
	/**
	 * Like writeArchive, but instead of writing to a file, stores the data in memory in the
	 * provided Archive::Builder.
	 */
	template<typename Ref, typename T = References<Ref>>
	Promise<void> buildArchive(Ref reference, Archive::Builder out, Maybe<Nursery&> nursery = Maybe<Nursery&>()) KJ_WARN_UNUSED_RESULT;
	
	//! Publish the data in the given Archive::Reader as LocalDataRef
	/**
	 * Like publishArchive, but copies the data from the in-memory structure given
	 */
	template<typename T>
	LocalDataRef<T> publishArchive(Archive::Reader in);
	
	///@}
	
	//! \name Limit configuration
	///@{
		
	struct Limits {
		uint64_t maxRAMObjectSize = 100000000; // Store up to 100MB in Ram
		Maybe<uint64_t> ramRemaining = nullptr; // Sets the remaining RAM budget. After this amount of RAM is requested, switch to file-backed allocation
	};
	
	void setLimits(Limits limits);	
	
	///@}
	
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
template<typename T>
class LocalDataRef : public DataRef<T>::Client {
public:
	using typename DataRef<T>::Client::Calls;
	
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

	ArrayPtr<const byte> getID();
	ArrayPtr<capnp::Capability::Client> getCapTable();
	uint64_t getTypeID();

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
	
	Temporary(Temporary<T>&&) = default;
	Temporary<T>& operator=(Temporary<T>&& other) = default;
	
	Temporary<T>& operator=(typename T::Reader other) {
		return (*this) = Temporary<T>(other);
	}
	
	typename T::Builder asBuilder() { return *this; }
	
	operator capnp::MessageBuilder&() { return *holder; }
	
	Own<capnp::MallocMessageBuilder> holder;
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

template<typename Key, typename T, template<typename, typename> typename Map = kj::TreeMap>
struct Cache {
	struct Holder;
	struct Ref;
	
	struct InsertResult {
		T& element;
		Ref ref;
	};
	
	InsertResult insert(Key key, T t);
	Maybe<T&> find(Key key);
	
	Map<Key, Own<Holder>> map;
};

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


}

#include "data-inl.h"
