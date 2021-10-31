#include "store.h"

namespace fsc {

inline int internal::compareDSKeys(const kj::ArrayPtr<const byte>& k1, const kj::ArrayPtr<const byte>& k2) {
	if(k1.size() < k2.size())
		return -1;
	else if(k1.size() > k2.size())
		return 1;
	
	return memcmp(k1.begin(), k2.begin(), k1.size());
}

kj::Own<const LocalDataStore::Entry> LocalDataStore::insertOrGet(const kj::ArrayPtr<const byte>& key, kj::Array<const byte>&& data) {
	return table.findOrCreate(key, [&]() { return kj::atomicRefcounted<Entry>(key, kj::mv(data)); }) -> atomicAddRef();
}

LocalDataStore::Entry::Entry(
	const kj::ArrayPtr<const byte>& key,
	kj::Array<const byte>&& data
) :
	key(kj::heapArray(key)),
	value(kj::mv(data))
{}

DataStoreMessageReader::DataStoreMessageReader(
	kj::Own<const LocalDataStore::Entry>&& entry,
	const capnp::ReaderOptions& options
) : 
	FlatArrayMessageReader(
		kj::ArrayPtr<const capnp::word>(reinterpret_cast<const capnp::word*>(entry->value.begin()), entry->value.size() / sizeof(capnp::word)),
		options
	),
	entry(kj::mv(entry))
	{}

kj::Own<DataStoreMessageReader> DataStoreMessageReader::copy() {
	return kj::heap<DataStoreMessageReader>(entry->atomicAddRef(), getOptions());
}

}