#include "capi-store.h"

namespace fsc {

struct StoreEntry {
	StoreEntry(fusionsc_DataStoreEntry*);
	~StoreEntry();
	
	StoreEntry(const StoreEntry&);
	StoreEntry(StoreEntry&&);
	
	StoreEntry& operator=(const StoreEntry&);
	StoreEntry& operator=(StoreEntry&&);
	
	fusionsc_DataStoreEntry* incRef();
	fusionsc_DataStoreEntry* release();
	
	kj::ArrayPtr<const byte> asPtr();
	kj::Array<const byte> asArray();

private:
	fusionsc_DataStoreEntry* raw = nullptr;
};

struct DataStore {
	DataStore(fusionsc_DataStore* store);
	~DataStore();
	
	DataStore(const DataStore&);
	DataStore(DataStore&&);
	
	DataStore& operator=(const DataStore&);
	DataStore& operator=(DataStore&&);
	
	fusionsc_DataStore* incRef();
	fusionsc_DataStore* release();
	
	StoreEntry publish(ArrayPtr<const byte> key, Array<const byte> data);
	Maybe<StoreEntry> query(ArrayPtr<const byte> key);
	
	void gc();
	
private:
	fusionsc_DataStore* raw = nullptr;
};

DataStore createStore();

}