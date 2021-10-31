#pragma once

#include <kj/map.h>
#include <kj/refcount.h>
#include <kj/array.h>
#include <kj/async.h>

#include <capnp/serialize.h>

#include "common.h"

namespace fsc {

/**
 * Central access point to share large binary data sections with each other. Organized as a table
 * of rows which each have an atomic reference counter. Usually once should access the shared
 * instance of the class through the thread- or library-handle.
 *
 * \code{.cpp}
 * Library l;
 * ThreadHandle h(l);
 * 
 * {
 *   kj::Locked<LocalDataStore> lds = l.store.lockExclusive();
 * }
 * {
 * 	 kj::Locked<LocalDataStore> lds = h.store().lockExclusive();
 * }
 * \endcode
 * 
 */
class LocalDataStore {
public:
	class Entry;
	
	using Key = kj::ArrayPtr<const byte>;
	using Row = kj::Own<const Entry>;
	
	class TreeIndexCallbacks {
	public:
		
		inline Key keyForRow (const Row& r) const;
		inline bool isBefore (const Row& r, Key k) const;
		inline bool matches  (const Row& r, Key k) const;
	};
	
	// Data stores are intended to hold data across threads, so we avoid unneccessarily moving data between nodes & worker threads.
	// Even though access to the store itself is usually mutex guarded, references to the data can be copied and removed
	// without any further notice. Therefore, the arrays need to be held in a container with shared refcounting.
	
	// Callbacks for tree index
	// See capnproto's kj/table.h for details
	//
	// TODO: Move this out of the header into the implementation file
	
	using Index = kj::TreeIndex<TreeIndexCallbacks>;
	using Table = kj::Table<Row, Index>;

	Table table;
	
	/**
	 * Looks up the key in the storage table and, if the row is not present,
	 * creates a new row and returns it. If the row is already present, the
	 * data array is discarded and the row is returned.
	 */
	kj::Own<const Entry> insertOrGet(const kj::ArrayPtr<const byte>& key, kj::Array<const byte>&& data);
};

/**
 * Entry row for the local data store. References to this row can be obtained by atomicAddRef
 * and destroyed without any synchronization requirements.
 */ 
class LocalDataStore::Entry : public kj::AtomicRefcounted {
public:
	inline Entry(const kj::ArrayPtr<const byte>& key, kj::Array<const byte>&& data);
	
	// Obtains a new reference to this entry
	inline kj::Own<      Entry> atomicAddRef()       { return kj::atomicAddRef(*this); }
	inline kj::Own<const Entry> atomicAddRef() const { return kj::atomicAddRef(*this); }
	
	kj::Array<const byte> key;
	kj::Array<const byte> value;
};

// NOTE: This class might get removed. A message reader that can be created directly from data store rows. 
class DataStoreMessageReader : public capnp::FlatArrayMessageReader {
public:
	// Creates a message reader from a data store row and, potentially, a set of reader options
	DataStoreMessageReader(kj::Own<const LocalDataStore::Entry>&& entry, const capnp::ReaderOptions& options = capnp::ReaderOptions());
	
	// Creates a new message reader on the same underlying data row (though with new state)
	kj::Own<DataStoreMessageReader> copy();
	
private:
	kj::Own<const LocalDataStore::Entry> entry;
};


namespace internal {
// Defines an ordering of byte arrays on data store keys.
int compareDSKeys(const kj::ArrayPtr<const byte>& k1, const kj::ArrayPtr<const byte>& k2);
}

inline LocalDataStore::Key LocalDataStore::TreeIndexCallbacks::keyForRow (const Row& r) const { return r ->key; }
inline bool LocalDataStore::TreeIndexCallbacks::isBefore (const Row& r, Key k) const { return internal::compareDSKeys(r -> key, k) < 0; }
inline bool LocalDataStore::TreeIndexCallbacks::matches  (const Row& r, Key k) const { return internal::compareDSKeys(r -> key, k) == 0; }

} // namespace fsc