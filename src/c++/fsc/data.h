#include <fsc/data.capnp.h>
#include <capnp/any.h>
#include <kj/async.h>
//#include <kj/filesystem.h>

#include "common.h"
#include "local.h"

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
}

// API forward declarations

template<typename T> class LocalDataRef;
class LocalDataService;

template<typename T>
struct TensorReader;

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
	Promise<LocalDataRef<T>> download(Reference src, bool recursive = true);
	
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
	 * Downloads the target data and all its transitive dependencies and writes them
	 * into an archive file. This file can then be shared with other customers to provide
	 * them a deep copy of the stored data.
	 */
	template<typename Ref, typename T = References<Ref>>
	Promise<void> writeArchive(Ref reference, const kj::File& out) KJ_WARN_UNUSED_RESULT;	
	
	/**
	 * Reads an archive file and publishes all data contained within. Returns a LocalDataRef
	 * to the root used when writing the archive.
	 */
	template<typename T>
	LocalDataRef<T> publishArchive(const kj::ReadableFile& in);
	
	/**
	 * Like writeArchive, but instead of writing to a file, stores the data in memory in the
	 * provided Archive::Builder.
	 */
	template<typename Ref, typename T = References<Ref>>
	Promise<void> buildArchive(Ref reference, Archive::Builder out) KJ_WARN_UNUSED_RESULT;
	
	/**
	 * Like publishArchive, but copies the data from the in-memory structure given
	 */
	template<typename T>
	LocalDataRef<T> publishArchive(Archive::Reader in);
	
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

template<typename T>
using TensorVal = decltype(instance<T>().getData().get(0));

/**
 * Reads a single element from an underlying tensor.
 *
 * WARNING: To enforce security on untrusted data, if the underlying class is a
 * ...::Reader, then capnproto performs a bounds check on the shape and data sections.
 * If you need to read repeatedly read from this tensor, use a TensorReader. This
 * will move the bounds check to its creation.
 *
 * TODO: Potential conversion to Eigen3 tensors.
 */
template<typename T>
TensorVal<T> tensorGet(const T& tensor, const ArrayPtr<size_t> index);

template<typename T>
void tensorSet(const T& tensor, const ArrayPtr<size_t> index, TensorVal<T> value);

template<typename T>
struct TensorReader {	
	TensorReader(const T ref);	
	TensorVal<T> get(const ArrayPtr<size_t> index);
	
	decltype(instance<const T>().getData()) data;
	decltype(instance<const T>().getShape()) shape;
};

}

#include "data-inl.h"
