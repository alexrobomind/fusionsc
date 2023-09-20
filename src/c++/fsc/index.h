#pragma once

#include "intervals.h"
#include "eigen.h"

#include <fsc/index.capnp.h>

namespace fsc {
	KDTreeService::Client newKDTreeService();
	
	template<int dims>
	struct KDTreeIndex {
		struct FindResult {
			double distance;
			uint64_t key;
		};
		
		inline CUPNP_FUNCTION FindResult findNearest(const Vec<double, dims>& x);
		
		inline CUPNP_FUNCTION KDTreeIndex(cu::KDTree::Reader tree) :
			tree(tree),
			chunkSplit(tree.getNTotal(), tree.getChunkSize())
		{}

	private:
		cu::KDTree::Reader tree;
		UnbalancedIntervalSplit chunkSplit;
		
		inline CUPNP_FUNCTION void findNearest(const Vec<double, dims>& x, FindResult& currentClosest, cu::KDTree::Node::Reader node);
		inline static Vec<double, dims> closestPoint(Vec<double, dims> x, cupnp::List<double>::Reader bounds);
		inline static Vec<double, dims> furthestPoint(Vec<double, dims> x, cupnp::List<double>::Reader bounds);
		
		inline double distance(const Vec<double, dims>& x1, const Vec<double, dims>& x2) {
			return (x1 - x2).dot(x1 - x2);
		}
	};
}

#include <fsc/index-inl.h>