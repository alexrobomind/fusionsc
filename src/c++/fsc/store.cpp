#include <list>

#include <kj/async-io.h>
#include <kj/function.h>
#include <kj/exception.h>

#include "store.h"

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

// === class LocalDataStore::Steward ===

LocalDataStore::Steward::Steward(kj::MutexGuarded<LocalDataStore>& target) :
	_store(target),
	_gcRequest(kj::newPromiseAndFulfiller<void>()),
	_thread([this](){ _run(); })
{}

LocalDataStore::Steward::~Steward() {
	getExecutor().executeSync([this](){
		_canceler.cancel(STEWARD_STOPPED);
	});
}

const kj::Executor& LocalDataStore::Steward::getExecutor() {
	auto locked = _executor.lockExclusive();
	locked.wait([](const Own<const kj::Executor>& exec) { return exec != nullptr; });
	return **locked;
}

void LocalDataStore::Steward::syncGC() {	
	// Perform garbage collection
	//std::list<Array<const byte>> orphans;
	
	{
		auto lStore = _store.lockExclusive();
		
		/*for(auto pRow = lStore->table.begin(); pRow != lStore->table.end(); ++pRow) {
			if((*pRow) -> isShared())
				continue;
			
			orphans.push_back(mv((*pRow) -> value));
		}*/
		
		auto predicate = [](Row& r) { return !r -> isShared(); };
		lStore->table.eraseAll(predicate);
	}
}

void LocalDataStore::Steward::asyncGC() {
	getExecutor().executeSync([this](){
		_gcRequest.fulfiller -> fulfill();
	});
}

void LocalDataStore::Steward::_run() {
	// Set up the event loop
	kj::AsyncIoContext aioCtx = kj::setupAsyncIo();
	
	// Store executor and gc request promise
	_gcRequest = kj::newPromiseAndFulfiller<void>();
	*(_executor.lockExclusive()) = kj::getCurrentThreadExecutor().addRef();
	
	kj::Function<Promise<void>()> loop = [this, &aioCtx, &loop]() {	
		syncGC();
		
		// Schedule next GC
		_gcRequest = kj::newPromiseAndFulfiller<void>();
		
		return _gcRequest.promise
		.exclusiveJoin(aioCtx.provider -> getTimer().afterDelay(30 * kj::SECONDS))
		.then(loop);
	};
	
	try {
		_canceler.wrap(kj::evalLater(loop)).wait(aioCtx.waitScope);
	} catch(kj::Exception& e) {
		if(e.getDescription() != STEWARD_STOPPED)
			throw e;
	}
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