#include "common.h"
#include "store.h"

#include <kj/table.h>

#include <atomic>
#include <cstdlib>

namespace fsc { namespace {

// LOCK ORDER (must be followed in this order to prevent deadlocks):
//   1. table (kj::MutexGuarded<Table>)
//   2. entry refcount (std::atomic<uint64_t>) via CAS operations
//   3. lruCache (kj::MutexGuarded<kj::List<...>>)
//
// IMPORTANT: When needing both table and lruCache locks, you MUST:
//   - First acquire table lock (shared or exclusive)
//   - Then acquire lruCache lock
//   - Release in reverse order (lruCache → table)
//
// The only exception is in gcImpl/publishImpl cache cleanup where we
// temporarily invert the order using tryClear() which "soft-locks" entries
// (refcount=1) without holding the table lock. This is safe because
// tryClear() only operates on entries that are already not in the table's
// active path (refcount=0 means no external references exist).

// THREAD SAFETY OVERVIEW:
// The DataStore implementation uses a combination of:
//   - Shared/exclusive locks on the global table (kj::MutexGuarded)
//   - Atomic reference counting with CAS (compare-and-swap) operations
//   - "Soft-locking" via refcount=1 to allow cache operations without table lock
//
// Lock acquisition rules:
//   - Always acquire locks in order: table → lruCache
//   - Use tryClear() for cache cleanup which soft-locks entries (refcount=1)
//   - If tryClear() returns false, another thread is modifying that entry
//   - CAS operations use default memory_order_seq_cst for full ordering
//
// Refcount state machine:
//   0 = deleted / can be reclaimed (entry not in use anywhere)
//   1 = locked for modification (only internal modifications happening)
//   2+ = active with external references (1 internal + N external refs)
//
// incRefImpl() will abort if refcount < 2, catching attempts to reference
// deleted entries. This is a safety feature, not a bug.

struct DataStoreImpl;
struct DataStoreEntryImpl;

using Key = kj::ArrayPtr<const byte>;
using Row = Own<DataStoreEntryImpl>;

struct TreeIndexCallbacks {	
	inline Key keyForRow (const Row& r) const;
	inline bool isBefore (const Row& r, Key k) const;
	inline bool matches  (const Row& r, Key k) const;
};

struct DataStoreEntryImpl : public fusionsc_DataStoreEntry {
	DataStoreImpl& parent;
	
	kj::ListLink<DataStoreEntryImpl> cacheLink;
	
	const ID key;
	
	// This field has 2 special values:
	//  1 -> not in use by clients, but locked for use by a method here in this file
	//  0 -> not in use anywhere and can be deleted
	//
	// REFCOUNT STATE MACHINE:
	//   0 = deleted / can be reclaimed (entry not in use anywhere)
	//   1 = locked for modification (only internal modifications happening)
	//   2+ = active with external references (1 internal + N external refs)
	//
	// State transitions:
	//   new entry -> refcount=0 (uninitialized, not in table yet)
	//   publish() finds existing -> refcount=2 (internal + first external)
	//   publish() creates new -> refcount=2 (internal + first external)
	//   incRef() on active entry -> refcount++
	//   decRefImpl() finds refcount=1 -> entry locked, will decide keep/delete
	//   decRefImpl() deletes backend -> refcount=0 (marks for deletion)
	//   decRefImpl() keeps backend -> refcount=0 then tryClear() or revive sets to 0
	//   tryRevive() succeeds -> refcount=2 (entry back in active use)
	//   tryClear() succeeds -> refcount=0 (entry marked for deletion)
	//
	// incRefImpl() ABORTS if refcount < 2, catching attempts to reference
	// deleted entries. This is a safety feature, not a bug.
	mutable std::atomic<uint64_t> refcount;
	// mutable std::atomic<fusionsc_DataHandle*> backend;
	mutable std::atomic<fusionsc_DataHandle*> backend;
	
	DataStoreEntryImpl(DataStoreImpl& parent, ArrayPtr<const byte> data);
	~DataStoreEntryImpl();
	
	void incRefImpl() const;
	void decRefImpl();
	
	void adopt(fusionsc_DataHandle* newHandle) noexcept;
	bool tryRevive();
	bool tryClear(); // Must be called with an exclusive lock on LRU cache held
};

struct DataStoreImpl : public fusionsc_DataStore {	
	using Index = kj::TreeIndex<TreeIndexCallbacks>;
	using Table = kj::Table<Row, Index>;
	
	std::atomic<size_t> refCount = 1;
	
	// LOCK ORDER: table -> object lock (refcount != 0) -> lruCache
	kj::MutexGuarded<Table> table;
	kj::MutexGuarded<kj::List<DataStoreEntryImpl, &DataStoreEntryImpl::cacheLink>> lruCache;
	
	DataStoreImpl();
	~DataStoreImpl();
	
	void incRefImpl();
	void decRefImpl(); // May lock lruCache
	
	fusionsc_DataStoreEntry* publishImpl(Key k, fusionsc_DataHandle* hdl);
	fusionsc_DataStoreEntry* queryImpl(Key k);
	void gcImpl();
	
	uint64_t cacheMin = 300000000;
	uint64_t cacheMax = 500000000;
	
	mutable std::atomic<uint64_t> currentSize = 0;
};
	
DataStoreEntryImpl::DataStoreEntryImpl(DataStoreImpl& parent, ArrayPtr<const byte> keyData) :
	parent(parent), refcount(0), backend(nullptr), key(kj::heapArray(keyData))
{
	dataPtr = nullptr;
	dataSize = 0;
	
	keyPtr = key.data.begin();
	keySize = key.data.size();
	
	incRef = [](fusionsc_DataStoreEntry* pEntry) noexcept {
		static_cast<DataStoreEntryImpl*>(pEntry) -> incRefImpl();
	};
	decRef = [](fusionsc_DataStoreEntry* pEntry) noexcept {
		static_cast<DataStoreEntryImpl*>(pEntry) -> decRefImpl();
	};
}

DataStoreEntryImpl::~DataStoreEntryImpl() {
	// Clean up backend if present
	if(backend != nullptr) {
		auto val = backend.load();
		
		parent.currentSize -= val -> dataSize;
		val -> free(val);
	}
	// Note: We might still be linked in LRU cache if we're being destroyed
	// during DataStoreImpl destructor. The parent destructor clears the
	// LRU cache before this entry's destructor runs, so this check
	// will return false in that scenario.
	if(cacheLink.isLinked()) {
		parent.lruCache.lockExclusive() -> remove(*this);
	}
}

void DataStoreEntryImpl::incRefImpl() const {
	if(refcount < 2) {
		KJ_DBG("IncRef called on empty DataStore entry");
		std::abort();
	}
	++refcount;
}

void DataStoreEntryImpl::decRefImpl() {
	auto& savedParent = parent;
	
	// Check if we are the last reference
	if(--refcount > 1)
		return;
	// object is now locked (refcount == 1 means no external refs, only internal mods)
	
	// Refcount should now be 1
	if(refcount != 1) {
		KJ_DBG("Inconsistent reference count in store");
		std::abort();
	}
	
	// NOTE: At this point, we own the entry exclusively. The entry may be:
	//   - Still in LRU cache (if it was evicted and kept for caching)
	//   - Not in LRU cache (if it was actively used)
	// We must remove it from LRU cache before deciding keep/delete.
	
	if(backend == nullptr) {
		KJ_DBG("Data store entry active with qnullptr content");
		std::abort();
	}
	
	auto val = backend.load();
	
	// Make sure we are not in LRU cache
	// (we might have been revived while in LRU, so this is race-safe)
	if(cacheLink.isLinked()) {
		auto lockedCache = savedParent.lruCache.lockExclusive();
		lockedCache -> remove(*this);
	}
	
	// Decide whether we want to keep the object alive or delete it
	if(savedParent.currentSize >= savedParent.cacheMax) {
		// Delete it - entry will be marked for deletion
		backend = nullptr;
		savedParent.currentSize -= val -> dataSize;
	} else {
		// Keep it - add to LRU cache for potential eviction later
		val = nullptr;  // Release our reference to val
		KJ_DBG("Keeping object in cache");
		auto lockedCache = savedParent.lruCache.lockExclusive();
		lockedCache -> add(*this);
	}
	
	// Mark object for deletion
	// After setting refcount=0, no other thread can incRef this entry.
	// We cannot access *this* after this point if the parent was also deleted.
	refcount = 0;
	
	// object is now unlocked (other threads can now operate on entry again)
	
	if(val != nullptr) {
		val -> free(val);
	}
	
	// Cleanup former references held by this object
	// This may delete the parent (and therefore this entry) if refcount reaches 0.
	// However, entry objects are heap-allocated via kj::heap<>, so they persist
	// independently of parent deletion. The parent destructor clears LRU cache
	// before freeing memory, so cacheLink.isLinked() will be false during cleanup.
	savedParent.decRefImpl();
}

// Only safe under read lock of global table, so that it doesn't
// race with the cleanup procedure (gcImpl, tryClear).
//
// THREAD SAFETY: This function uses CAS-based locking on refcount:
//   - If refcount > 1: entry has external refs, we just increment
//   - If refcount == 1: entry is being modified by another thread, wait
//   - If refcount == 0: entry is deleted, this should not happen (abort in incRef)
//   - CAS to refcount=1 locks the entry for modification
//   - After modification, refcount=2 marks entry as active (1 internal + 1 new external)
void DataStoreEntryImpl::adopt(fusionsc_DataHandle* newHandle) noexcept {
	auto rcVal = refcount.load();
	
	while(true) {
		// refcount == 1 indicates another thread is modifying this entry
		// (adopt/decRefImpl/revive in progress). Spin until it completes.
		while(rcVal == 1) {
			rcVal = refcount.load();
		}
		
		// Branch for already active handle (refcount >= 2)
		if(rcVal > 0) {
			// Register new reference atomically
			if(!refcount.compare_exchange_weak(rcVal, rcVal + 1))
				continue;
			newHandle -> free(newHandle);  // Handle already exists, discard ours
			return;
		}
		
		// Handle is actually not in use, steal it for modification
		// CAS to refcount=1 locks the entry exclusively
		if(!refcount.compare_exchange_weak(rcVal, 1))
			continue;
		
		// object is now locked (exclusive access)
		parent.incRefImpl();  // Register new reference on parent
		
		// Clean up old backend if present
		if(backend != nullptr) {
			auto val = backend.load();
			val -> free(val);
			parent.currentSize -= val -> dataSize;
		}
		
		// Remove from LRU cache if present (was revived from cache)
		if(cacheLink.isLinked()) {
			auto lockedCache = parent.lruCache.lockExclusive();
			lockedCache -> remove(*this);
		}
		
		// Install new backend
		backend = newHandle;
		dataPtr = newHandle -> dataPtr;
		dataSize = newHandle -> dataSize;
		parent.currentSize += dataSize;
		
		// Mark entry as active: 1 internal ref + 1 external ref
		refcount = 2;
		// object is now unlocked
		return;
	}
}

// Only safe under read lock of global table, so that it doesn't
// race with the cleanup procedure.
//
// REVIVE MECHANISM:
// An entry in the LRU cache has backend!=nullptr but refcount=0.
// tryRevive() makes it usable again by:
//   1. CAS lock (refcount=1) to prevent concurrent modifications
//   2. Verify backend is still non-null
//   3. Increment refcount (refcount=2) and remove from LRU
//   4. Return true for success
//
// RACE CONDITION NOTES:
// - The initial backend==nullptr check at line 228 is not atomic
// - If another thread sets backend=nullptr between our check and CAS,
//   we detect it at line 319 and return false safely
// - If another thread increments refcount while we hold lock (refcount=1),
//   our CAS fails and we retry
bool DataStoreEntryImpl::tryRevive() {
	// Quick check without atomics (may have race but it's benign)
	if(backend == nullptr)
		return false;
	
	auto rcVal = refcount.load();
	
	while(true) {
		// Another thread is modifying this entry, wait for it
		while(rcVal == 1) {
			rcVal = refcount.load();
		}
		
		// Entry is active (has external refs), just add ours
		if(rcVal > 0) {
			if(!refcount.compare_exchange_weak(rcVal, rcVal + 1))
				continue;
			return true;
		}
		
		// Lock the entry for modification
		if(!refcount.compare_exchange_weak(rcVal, 1))
			continue;
		
		// object is now locked
		
		if(backend != nullptr) {
			parent.incRefImpl();
			
			// Can successfully revive. Remove from LRU cache.
			if(cacheLink.isLinked()) {
				auto lockedCache = parent.lruCache.lockExclusive();
				lockedCache -> remove(*this);
			}
			
			refcount = 2;  // 1 internal + 1 external
			// object is now unlocked
			return true;
		} else {
			// Entry has its content cleared out
			refcount = 0;
			// object is now unlocked
			return false;
		}
	}
}

// Only safe under read lock of global table, so that it doesn't
// race with the cleanup procedure.
//
// Soft-locks an entry and clears its backend if possible.
// Returns false if the entry is currently being modified.
//
// CALLED FROM: publishImpl() cache cleanup section (line 421)
// LOCK REQUIREMENT: Caller MUST hold lruCache.lockExclusive() when calling.
// This is an "inverted lock order" pattern: we acquire cache lock first,
// then modify entries without holding table lock.
//
// Soft-locking (refcount=1) prevents other threads from:
//   - IncRef'ing the entry (would abort per incRefImpl check)
//   - Reviving the entry (CAS loop would see refcount=1)
//   - Clearing the entry (CAS would fail)
//
// If tryClear() returns false, the entry was being modified by another
// thread. Release the cache lock, wait briefly, and retry.
bool DataStoreEntryImpl::tryClear() {
	uint64_t desired = 0;
	if(!refcount.compare_exchange_weak(desired, 1))
		return false;
	
	// object is now locked
	
	// Remove from LRU cache if present
	if(cacheLink.isLinked()) {
		auto& lockedCache = parent.lruCache.getAlreadyLockedExclusive();
		lockedCache.remove(*this);
	}
	
	if(backend != nullptr) {
		auto val = backend.load();
		
		parent.currentSize -= val -> dataSize;
		val -> free(val);
		
		backend = nullptr;
	}
	
	refcount = 0;
	// object is now unlocked
	
	return true;
}

inline int compareDSKeys(const Key& k1, const Key& k2) {
	if(k1.size() < k2.size())
		return -1;
	else if(k1.size() > k2.size())
		return 1;
	
	return memcmp(k1.begin(), k2.begin(), k1.size());
}

Key TreeIndexCallbacks::keyForRow (const Row& r) const {
	return r -> key.data;
}

bool TreeIndexCallbacks::isBefore (const Row& r, Key k) const {
	return r -> key < k;
}
bool TreeIndexCallbacks::matches  (const Row& r, Key k) const {
	return r -> key == k;
}

DataStoreImpl::DataStoreImpl() {
	incRef = [](fusionsc_DataStore* self) noexcept {
		static_cast<DataStoreImpl*>(self) -> incRefImpl();
	};
	decRef = [](fusionsc_DataStore* self) noexcept {
		static_cast<DataStoreImpl*>(self) -> decRefImpl();
	};
	publish = [](fusionsc_DataStore* self, const unsigned char* keyData, size_t keySize, fusionsc_DataHandle* data) noexcept {
		return static_cast<DataStoreImpl*>(self) -> publishImpl(Key(keyData, keySize), data);
	};
	query = [](fusionsc_DataStore* self, const unsigned char* keyData, size_t keySize) noexcept {
		return static_cast<DataStoreImpl*>(self) -> queryImpl(Key(keyData, keySize));
	};
	gc = [](fusionsc_DataStore* self) noexcept {
		static_cast<DataStoreImpl*>(self) -> gcImpl();
	};
}

DataStoreImpl::~DataStoreImpl() {
	// Clear LRU cache before freeing memory.
	// Entry destructors (DataStoreEntryImpl::~DataStoreEntryImpl) remove
	// themselves from the LRU cache, so this loop safely clears all entries.
	// After this, cacheLink.isLinked() will be false for all entries.
	{
		auto locked = lruCache.lockExclusive();
		
		while(!locked -> empty()) {
			locked -> remove(locked -> front());
		}
	}
	// LRU cache is now empty. Entry destructors will see cacheLink.isLinked()==false
}
	
void DataStoreImpl::incRefImpl() {
	++refCount;
}

void DataStoreImpl::decRefImpl() {
	if(--refCount == 0)
		delete this;
}

fusionsc_DataStoreEntry* DataStoreImpl::publishImpl(Key k, fusionsc_DataHandle* hdl) {
	fusionsc_DataStoreEntry* result = nullptr;
	
	// First try to adopt existing row (shared lock for fast path)
	{
		auto locked = table.lockShared();
		
		KJ_IF_MAYBE(pRow, locked -> find(k)) {
			// We know we can const-cast this away because the table itself is mutable,
			// and the per-row access synchronization is handled through the refCount guard
			const DataStoreEntryImpl* dstReadonly = pRow -> get();
			DataStoreEntryImpl* dst = const_cast<DataStoreEntryImpl*>(dstReadonly);
			
			dst -> adopt(hdl);  // This also calls incRef
			result = dst;
		}
	}  // table shared lock released here
	
	// Note: If we found an entry above, it's possible that between the
	// shared lock release and our return, another thread could call
	// decRefImpl and decide to delete the entry. However, since we've
	// already called adopt() which increments refcount, the entry will
	// either be kept (refcount=2 after adopt) or the caller gets a valid
	// reference. The entry can only be deleted when refcount reaches 0.
	
	// Now try again with locked table (entry didn't exist)
	if(result == nullptr) {
		auto locked = table.lockExclusive();
		Row& row = locked -> findOrCreate(
			k,
			[this, k]() {
				return kj::heap<DataStoreEntryImpl>(*this, k);
			}
		);
		row -> adopt(hdl);
		result = row.get();
	}
	
	// Downsize the cache if it hits a maximum value
	if(currentSize > cacheMax) {
		// This is necessary to prevent the GC from eating rows while we
		// are operating on them.
		auto lockedTable = table.lockShared();
		
		// Lock order inversion: We need table -> cache -> object,
		// but the global order is table -> object -> cache.
		// Solution: tryClear() "soft-locks" entries (refcount=1) which
		// prevents other threads from reviving them. If tryClear() fails
		// (returns false), another thread is modifying that entry, so we
		// release cache lock, wait, and retry.
		
		while(currentSize > cacheMin) {
			auto lockedCache = lruCache.lockExclusive();
			
			if(lockedCache -> empty())
				break;
			
			while(!lockedCache -> empty()) {
				DataStoreEntryImpl& entry = lockedCache -> front();
				auto blocked = !entry.tryClear();
				
				if(blocked)
					break;  // Entry was being modified, release cache lock
			}
			// Note: lockedCache is released at end of loop iteration
		}
	}
	
	return result;
}

fusionsc_DataStoreEntry* DataStoreImpl::queryImpl(Key k) {
	const auto locked = table.lockShared();
	KJ_IF_MAYBE(pRow, locked  -> find(k)) {
		auto& row = **pRow;
		uint64_t prevCount = row.refcount.load();
		
		while(true) {
			// Entry is deleted (refcount=0). Try to revive it from LRU cache.
			if(prevCount == 0) {
				auto pMutable = const_cast<DataStoreEntryImpl*>(&row);
				
				KJ_DBG("Pre-existing row");
				
				if(pMutable -> tryRevive()) {
					KJ_DBG("Revived from cache");
					return pMutable;
				}
				
				// Entry was cleared and can't be revived
				return nullptr;
			}
			
			// Entry is active, try to increment refcount atomically
			if(row.refcount.compare_exchange_weak(prevCount, prevCount + 1))
				return const_cast<fusionsc_DataStoreEntry*>(static_cast<const fusionsc_DataStoreEntry*>(&row));
			// prevCount was updated by CAS to current value, retry
		}
	}
	return nullptr;
}

void DataStoreImpl::gcImpl() {
	auto locked = table.lockExclusive();
	locked -> eraseAll([](Row& r) -> bool {
		return r -> refcount == 0 && r -> backend == nullptr;
	});
}

}

// class StoreEntry

StoreEntry::StoreEntry(fusionsc_DataStoreEntry* newBackend) :
	raw(newBackend)
{}

StoreEntry::~StoreEntry()
{
	if(raw != nullptr) {
		raw -> decRef(raw);
	}
}

StoreEntry::StoreEntry(StoreEntry&& other) :
	raw(other.raw)
{
	other.raw = nullptr;
}

StoreEntry::StoreEntry(const StoreEntry& other) :
	raw(other.raw)
{
	if(raw != nullptr)
		raw -> incRef(raw);
}

StoreEntry& StoreEntry::operator=(StoreEntry&& other) {
	if(raw != nullptr)
		raw -> decRef(raw);
	
	raw = other.raw;
	other.raw = nullptr;
	
	return *this;
}

StoreEntry& StoreEntry::operator=(const StoreEntry& other) {
	auto tmp = raw;
	raw = other.raw;
	
	if(raw != nullptr)
		raw -> incRef(raw);
	
	if(tmp != nullptr)
		tmp -> decRef(tmp);
	
	return *this;
}

ArrayPtr<const byte> StoreEntry::asPtr() {
	return ArrayPtr<const byte>(raw -> dataPtr, raw -> dataSize);
}

Array<const byte> StoreEntry::asArray() {
	return asPtr().attach(cp(*this));
}

fusionsc_DataStoreEntry* StoreEntry::incRef() {
	raw -> incRef(raw);
	return raw;
}

fusionsc_DataStoreEntry* StoreEntry::release() {
	auto tmp = raw;
	raw = nullptr;
	return tmp;
}

// class DataStore


DataStore::DataStore(fusionsc_DataStore* newBackend) :
	raw(newBackend)
{
	if(raw == nullptr)
		raw = new DataStoreImpl();
	else
		raw -> incRef(raw);
}

DataStore::~DataStore()
{
	if(raw != nullptr) {
		raw -> decRef(raw);
	}
}

DataStore::DataStore(DataStore&& other) :
	raw(other.raw)
{
	other.raw = nullptr;
}

DataStore::DataStore(const DataStore& other) :
	raw(other.raw)
{
	if(raw != nullptr)
		raw -> incRef(raw);
}

DataStore& DataStore::operator=(DataStore&& other) {
	if(raw != nullptr)
		raw -> decRef(raw);
	
	raw = other.raw;
	other.raw = nullptr;
	
	return *this;
}

DataStore& DataStore::operator=(const DataStore& other) {
	auto tmp = raw;
	raw = other.raw;
	
	if(raw != nullptr)
		raw -> incRef(raw);
	
	if(tmp != nullptr)
		tmp -> decRef(tmp);
	
	return *this;
}

fusionsc_DataStore* DataStore::incRef() {
	raw -> incRef(raw);
	return raw;
}

fusionsc_DataStore* DataStore::release() {
	auto tmp = raw;
	raw = nullptr;
	return tmp;
}

StoreEntry DataStore::publish(ArrayPtr<const byte> key, Array<const byte> data) {
	struct DataHandle : public fusionsc_DataHandle {
		Array<const byte> data;
		
		DataHandle(Array<const byte>&& newData) :
			data(mv(newData))
		{
			dataPtr = data.begin();
			dataSize = data.size();
			free = [](fusionsc_DataHandle* self) noexcept {
				delete static_cast<DataHandle*>(self);
			};
		}
	};
	
	auto hdl = new DataHandle(mv(data));
	return raw -> publish(raw, key.begin(), key.size(), hdl);
}

Maybe<StoreEntry> DataStore::query(ArrayPtr<const byte> key) {
	auto pResult = raw -> query(raw, key.begin(), key.size());
	if(pResult != nullptr)
		return StoreEntry(pResult);
	
	return nullptr;
}

void DataStore::gc() {
	raw -> gc(raw);
}

DataStore createStore() {
	return DataStore(nullptr);
}

}
