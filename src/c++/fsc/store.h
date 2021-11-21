#pragma once

#include <kj/map.h>
#include <kj/refcount.h>
#include <kj/array.h>
#include <kj/async.h>
#include <kj/thread.h>

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
	
	class Steward;
	
	// Data stores are intended to hold data across threads, so we avoid unneccessarily moving data between nodes & worker threads.
	// Even though access to the store itself is usually mutex guarded, references to the data can be copied and removed
	// without any further notice. Therefore, the arrays need to be held in a container with shared refcounting.
	
	// Callbacks for tree index
	// See capnproto's kj/table.h for details
	//
	// TODO: Move this out of the header into the implementation file
	
	using Index = kj::TreeIndex<TreeIndexCallbacks>;
	using Table = kj::Table<Row, Index>;
	
	/**
	 * Looks up the key in the storage table and, if the row is not present,
	 * creates a new row and returns it. If the row is already present, the
	 * data array is discarded and the row is returned.
	 */
	kj::Own<const Entry> insertOrGet(const kj::ArrayPtr<const byte>& key, kj::Array<const byte>&& data);

	Table table;
};

/**
 * Entry row for the local data store. References to this row can be obtained by atomicAddRef
 * and destroyed without any synchronization requirements.
 */ 
class LocalDataStore::Entry : public kj::AtomicRefcounted {
public:
	Entry(const kj::ArrayPtr<const byte>& key, kj::Array<const byte>&& data);
	
	// Obtains a new reference to this entry
	inline kj::Own<      Entry> addRef()       { return kj::atomicAddRef(*this); }
	inline kj::Own<const Entry> addRef() const { return kj::atomicAddRef(*this); }
	
	//kj::Array<const byte> key;
	ID key;
	kj::Array<const byte> value;
};

class LocalDataStore::Steward {
public:
	Steward(kj::MutexGuarded<LocalDataStore>& store);
	~Steward();
	
	const kj::Executor& getExecutor();
	
	// Synchronously runs the GC and then returns
	void syncGC();
	
	// Shift the next GC scheduled in the worker thread to "now"
	void asyncGC();

private:
	kj::MutexGuarded<LocalDataStore>& _store;
	kj::Canceler _canceler;
	
	kj::PromiseFulfillerPair<void> _gcRequest;
	kj::Thread _thread;
	
	kj::MutexGuarded<Own<const kj::Executor>> _executor;
	
	void _run();
};

// NOTE: This class might get removed. A message reader that can be created directly from data store rows. 
class DataStoreMessageReader : public capnp::FlatArrayMessageReader {
public:
	// Creates a message reader from a data store row and, potentially, a set of reader options
	DataStoreMessageReader(kj::Own<const LocalDataStore::Entry>&& entry, const capnp::ReaderOptions& options = capnp::ReaderOptions());
	
	// Creates a new message reader on the same underlying data row (though with new state)
	kj::Own<DataStoreMessageReader> addRef() const;
	
private:
	kj::Own<const LocalDataStore::Entry> entry;
	capnp::ReaderOptions options;
};


namespace internal {
// Defines an ordering of byte arrays on data store keys.
int compareDSKeys(const kj::ArrayPtr<const byte>& k1, const kj::ArrayPtr<const byte>& k2);
}

inline LocalDataStore::Key LocalDataStore::TreeIndexCallbacks::keyForRow (const Row& r) const { return r -> key; }
inline bool LocalDataStore::TreeIndexCallbacks::isBefore (const Row& r, Key k) const { return r -> key < k; }
inline bool LocalDataStore::TreeIndexCallbacks::matches  (const Row& r, Key k) const { return r -> key == k; }

} // namespace fsc