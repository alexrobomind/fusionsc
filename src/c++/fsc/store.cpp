#include "common.h"
#include "store.h"

#include <kj/table.h>

#include <atomic>
#include <cstdlib>

namespace fsc { namespace {

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
	mutable std::atomic<uint64_t> refcount;
	// mutable std::atomic<fusionsc_DataHandle*> backend;
	mutable std::atomic<fusionsc_DataHandle*> backend;
	
	DataStoreEntryImpl(DataStoreImpl& parent, ArrayPtr<const byte> data);
	~DataStoreEntryImpl();
	
	void incRefImpl() const;
	void decRefImpl();
	
	void adopt(fusionsc_DataHandle* newHandle);
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
	if(backend != nullptr) {
		auto val = backend.load();
		
		parent.currentSize -= val -> dataSize;
		val -> free(val);
	}
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
	// object is now locked
		
	// Refcount should now be 1
	if(refcount != 1) {
		KJ_DBG("Inconsistent reference count in store");
		std::abort();
	}
	
	if(backend == nullptr) {
		KJ_DBG("Data store entry active with qnullptr content");
		std::abort();
	}
	
	auto val = backend.load();
	
	// Make sure we are not in LRU cache
	if(cacheLink.isLinked()) {
		auto& lockedCache = savedParent.lruCache.lockExclusive();
		lockedCache -> remove(*this);
	}
	
	// Decide whether we want to keep the object alive
	// or delete it
	if(savedParent.currentSize >= savedParent.cacheMax) {
		// Delete it
		backend = nullptr;
		savedParent.currentSize -= val -> dataSize;
	} else {
		// Keep it
		val = nullptr;
		
		KJ_DBG("Keeping object in cache");
		
		// Add to LRU cache in case we run low on space
		auto& lockedCache = savedParent.lruCache.lockExclusive();
		lockedCache -> add(*this);
	}
	
	// Mark object for deletion
	// We can not access *this* after setting refcount to 0.
	refcount = 0;
	
	// object is now unlocked
	
	if(val != nullptr) {
		val -> free(val);
	}
	
	// Cleanup former references held by this object
	savedParent.decRefImpl(); // Note: < This can cause the parent (and therefore this) to get deleted.
}

// Only safe under read lock of global table, so that it doesn't
// race with the cleanup procedure.
void DataStoreEntryImpl::adopt(fusionsc_DataHandle* newHandle) {
	auto rcVal = refcount.load();
	
	while(true) {
		// refcount == 1 indicates that there are other
		// adopt / decRefImpl / revive calls concurrently modifying
		// this object.
		while(rcVal == 1) {
			rcVal = refcount.load();
		}
		
		// Branch for already active handle
		if(rcVal > 0) {
			// Register new reference
			if(!refcount.compare_exchange_weak(rcVal, rcVal + 1))
				continue;
				
			newHandle -> free(newHandle);
			return;
		}
		
		// Handle is actually not in use, steal it for modification
		if(!refcount.compare_exchange_weak(rcVal, 1))
			continue;
		
		// object is now locked
		parent.incRefImpl();
		
		if(backend != nullptr) {
			auto val = backend.load();
			
			val -> free(val);
			parent.currentSize -= val -> dataSize;
		}
	
		// Remove from LRU cache
		if(cacheLink.isLinked()) {
			auto& lockedCache = parent.lruCache.lockExclusive();
			lockedCache -> remove(*this);
		}
		
		backend = newHandle;
		dataPtr = newHandle -> dataPtr;
		dataSize = newHandle -> dataSize;
		
		parent.currentSize += dataSize;
		
		refcount = 2;
		// object is now unlocked
		return;
	}
}

// Only safe under read lock of global table, so that it doesn't
// race with the cleanup procedure.
bool DataStoreEntryImpl::tryRevive() {
	// Quick check without atomics
	if(backend == nullptr)
		return false;
	
	auto rcVal = refcount.load();
	
	while(true) {
		// refcount == 1 indicates that there are other
		// adopt / decRefImpl / revive calls concurrently modifying
		// this object.
		while(rcVal == 1) {
			rcVal = refcount.load();
		}
		
		// Branch for already active handle
		if(rcVal > 0) {
			// Register new reference
			if(!refcount.compare_exchange_weak(rcVal, rcVal + 1))
				continue;
				
			return true;
		}
		
		// Handle is actually not in use, steal it for modification
		if(!refcount.compare_exchange_weak(rcVal, 1))
			continue;
		
		// object is now locked
		
		if(backend != nullptr) {
			parent.incRefImpl();
			
			// Can successfully revive. Remove object from LRU cache
			if(cacheLink.isLinked()) {
				auto& lockedCache = parent.lruCache.lockExclusive();
				lockedCache -> remove(*this);
			}
		
			refcount = 2;
			// object is now unlocked
			
			return true;
		} else {
			refcount = 0;
			// object is now unlocked
			
			return false;
		}
		
	}
}

// Only safe under read lock of global table, so that it doesn't
// race with the cleanup procedure.
bool DataStoreEntryImpl::tryClear() {		
	uint64_t desired = 0;
	if(!refcount.compare_exchange_weak(desired, 1))
		return false;
	
	// object is now locked
	
	// Remove from LRU cache
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
	{
		auto locked = lruCache.lockExclusive();
		
		while(!locked -> empty()) {
			locked -> remove(locked -> front());
		}
	}
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
	
	// First try to adopt existing row
	{
		auto locked = table.lockShared();
		
		KJ_IF_MAYBE(pRow, locked -> find(k)) {
			
			// We know we can const-cast this away because the table itself is mutable,
			// and the per-row access synchronization is handled through the refCount guard
			const DataStoreEntryImpl* dstReadonly = pRow -> get();
			DataStoreEntryImpl* dst = const_cast<DataStoreEntryImpl*>(dstReadonly);
			
			dst -> adopt(hdl); // This also calls incRef
			result = dst;
		}
	}
	
	// Now try again with locked table
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
		// This is neccessary to prevent the GC from eating rows while we
		// are operating on them.
		auto lockedTable = table.lockShared();
		
		// Because we need to iterate the cache and then process objects,
		// we need to walk it with an inverted lock order. The global lock order
		// is table -> object -> cache, but we need to proceed table -> cache -> object.
		//
		// To deal with this, tryClear "soft-locks" the object, and if it fails to do so,
		// will return "false". I this case, we need to release the cache lock, because
		// another thread is currently trying to revive this object, and trying to remove
		// it from the LRU cache.
		
		while(currentSize > cacheMin) {
			auto lockedCache = lruCache.lockExclusive();
			
			if(lockedCache -> empty())
				break;
			
			while(!lockedCache -> empty()) {
				DataStoreEntryImpl& entry = lockedCache -> front();
				auto blocked = !entry.tryClear();
				
				if(blocked)
					break;
			}
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
			if(prevCount == 0) {
				// auto pMutable = const_cast<fusionsc_DataStoreEntry*>(static_cast<const fusionsc_DataStoreEntry*>(&row));
				auto pMutable = const_cast<DataStoreEntryImpl*>(&row);
				
				KJ_DBG("Pre-existing row");
				
				if(pMutable -> tryRevive()) {
					KJ_DBG("Revived from cache");
					return pMutable;
				}
				
				return nullptr;
			}
			
			if(row.refcount.compare_exchange_weak(prevCount, prevCount + 1))
				return const_cast<fusionsc_DataStoreEntry*>(static_cast<const fusionsc_DataStoreEntry*>(&row));
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
