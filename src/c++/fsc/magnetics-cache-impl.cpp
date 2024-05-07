#include "magnetics.h"

namespace fsc {

namespace {

struct LRUFieldCacheImpl : public FieldCache {
	using Key = kj::Tuple<kj::Array<const byte>, kj::Array<const byte>>;
	using KeyRef = kj::Tuple<kj::ArrayPtr<const byte>, kj::ArrayPtr<const byte>>;
	
	struct Row {
		kj::ListLink<Row> lruLink;
		
		Key key;
		ForkedPromise<LocalDataRef<Float64Tensor>> data;
		
		Row(kj::ArrayPtr<const byte> pointsHash, kj::ArrayPtr<const byte> fieldKey, Promise<LocalDataRef<Float64Tensor>>&& p) :
			key(kj::tuple(kj::heapArray(pointsHash), kj::heapArray(fieldKey))),
			data(p.fork())
		{}
	};
	
	struct TreeIndexCallbacks {
		int cmp(kj::ArrayPtr<const byte> a1, kj::ArrayPtr<const byte> a2) const {
			if(a1.size() < a2.size())
				return -1;
			if(a1.size() > a2.size())
				return 1;
			
			return memcmp(a1.begin(), a2.begin(), a1.size());
		}
		
		const Key& keyForRow(const Own<Row>& row) const { return row -> key; }
		bool isBefore(const Own<Row>& row, const KeyRef& key) const {
			auto cmp1 = cmp(kj::get<0>(row -> key), kj::get<0>(key));
			if(cmp1 < 0)
				return true;
			if(cmp1 > 0)
				return false;
			
			auto cmp2 = cmp(kj::get<1>(row -> key), kj::get<1>(key));
			return cmp2 < 0;
		}
		
		bool matches(const Own<Row>& row, const KeyRef& key) const {
			auto cmp1 = cmp(kj::get<0>(row -> key), kj::get<0>(key));
			if(cmp1 != 0)
				return false;
			
			auto cmp2 = cmp(kj::get<1>(row -> key), kj::get<1>(key));
			return cmp2 == 0;
		}
	};

	
	kj::List<Row, &Row::lruLink> lruQueue;
	kj::Table<Own<Row>, kj::TreeIndex<TreeIndexCallbacks>> table;
	
	unsigned int size = 0;
	
	LRUFieldCacheImpl(unsigned int size) :
		size(size)
	{}
	
	~LRUFieldCacheImpl() {
		for(auto& row : lruQueue)
			lruQueue.remove(row);
	}
	
	Maybe<Promise<LocalDataRef<Float64Tensor>>> check(kj::ArrayPtr<const byte> pointsHash, kj::ArrayPtr<const byte> fieldKey) override {
		KJ_IF_MAYBE(pEntry, table.find(kj::tuple(pointsHash, fieldKey))) {
			// Shuffle row to back
			auto& row = **pEntry;
			
			lruQueue.remove(row);
			lruQueue.add(row);
			
			// Return data copy
			return row.data.addBranch();
		}
		
		return nullptr;
	}
	
	void put(kj::ArrayPtr<const byte> pointsHash, kj::ArrayPtr<const byte> fieldKey, Promise<LocalDataRef<Float64Tensor>> newData) override {
		auto& pRow = table.insert(kj::heap<Row>(pointsHash, fieldKey, mv(newData)));
		auto& row = *pRow;
		
		lruQueue.add(row);
		
		if(lruQueue.size() > size) {
			auto& row = *lruQueue.begin();
			
			lruQueue.remove(*pRow);
			table.eraseMatch(row.key);
		}
	}
};

}

Own<FieldCache> lruFieldCache(unsigned int size) {
	return kj::heap<LRUFieldCacheImpl>(size);
}

Array<const byte> FieldCache::hashPoints(Eigen::TensorMap<Eigen::Tensor<double, 2>> points) {
	KJ_REQUIRE(points.dimension(1) == 3);
	
	auto hashFunc = Botan::HashFunction::create("Blake2b");
	hashFunc -> update((unsigned char*) points.data(), points.size() * sizeof(double));
	
	auto hash = kj::heapArray<uint8_t>(hashFunc -> output_length());
	hashFunc -> final(hash.begin());
	
	return hash.releaseAsBytes();
}

}