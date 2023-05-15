#include <list>

#include <kj/async-io.h>
#include <kj/function.h>
#include <kj/exception.h>

#include <exception>

#include "store.h"
#include "local.h"

namespace {
	kj::StringPtr STEWARD_STOPPED = "DataStore steward stopped"_kj;
}

namespace fsc {
	
// class LocalDataStore

kj::Own<const LocalDataStore::Entry> LocalDataStore::insertOrGet(const kj::ArrayPtr<const byte>& key, kj::Array<const byte>&& data) {
	return table.findOrCreate(key, [&]() { return kj::atomicRefcounted<Entry>(key, kj::mv(data)); }) -> addRef();
}

inline int internal::compareDSKeys(const kj::ArrayPtr<const byte>& k1, const kj::ArrayPtr<const byte>& k2) {
	if(k1.size() < k2.size())
		return -1;
	else if(k1.size() > k2.size())
		return 1;
	
	return memcmp(k1.begin(), k2.begin(), k1.size());
}

LocalDataStore::Entry::Entry(
	const kj::ArrayPtr<const byte>& key,
	kj::Array<const byte>&& data
) :
	key(kj::heapArray(key)),
	value(kj::mv(data))
{}

Promise<void> LocalDataStore::gcLoop(const kj::MutexGuarded<LocalDataStore>& store) {
	{
		auto lStore = store.lockExclusive();
		
		auto predicate = [](Row& r) { return !r -> isShared(); };
		lStore->table.eraseAll(predicate);
	}
	
	return getActiveThread().timer().afterDelay(1 * kj::SECONDS)
	.then([&store]() {
		return gcLoop(store);
	});
}

// === class DataStoreMessageReader ===

DataStoreMessageReader::DataStoreMessageReader(
	kj::Own<const LocalDataStore::Entry>&& entry,
	const capnp::ReaderOptions& options
) : 
	FlatArrayMessageReader(
		kj::ArrayPtr<const capnp::word>(reinterpret_cast<const capnp::word*>(entry->value.begin()), entry->value.size() / sizeof(capnp::word)),
		options
	),
	entry(kj::mv(entry)),
	options(options)
	{}

kj::Own<DataStoreMessageReader> DataStoreMessageReader::addRef() const {
	return kj::heap<DataStoreMessageReader>(entry->addRef(), options);
}

}