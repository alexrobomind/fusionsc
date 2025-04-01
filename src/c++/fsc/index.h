#pragma once

#include "intervals.h"
#include "eigen.h"

#include <fsc/index.capnp.h>
#include <fsc/index.capnp.cu.h>

namespace fsc {
	Own<KDTreeService::Server> newKDTreeService();
	
	struct KDTreeIndexBase {
		struct FindResult {
			double distance;
			uint64_t key;
		};
		
		struct NodeInfo {
			cu::KDTree::Node::Reader node;
			cupnp::List<double>::Reader bounds;
			
			inline CUPNP_FUNCTION double diameter();
			inline CUPNP_FUNCTION double diameterSqr();
		};
		
		inline CUPNP_FUNCTION KDTreeIndexBase(cu::KDTree::Reader tree) :
			tree(tree),
			chunkSplit(tree.getNTotal(), tree.getChunkSize())
		{}
		
		inline CUPNP_FUNCTION NodeInfo getNode(uint64_t id);
		
	protected:
		cu::KDTree::Reader tree;
		UnbalancedIntervalSplit chunkSplit;
	};
	
	template<int dims>
	struct KDTreeIndex : public KDTreeIndexBase {
		inline CUPNP_FUNCTION FindResult findNearest(const Vec<double, dims>& x);
		
		inline CUPNP_FUNCTION void findNearest(const Vec<double, dims>& x, FindResult& currentClosest, cu::KDTree::Node::Reader node);
		inline static Vec<double, dims> closestPoint(Vec<double, dims> x, cupnp::List<double>::Reader bounds);
		inline static Vec<double, dims> furthestPoint(Vec<double, dims> x, cupnp::List<double>::Reader bounds);
		
		inline double distance(const Vec<double, dims>& x1, const Vec<double, dims>& x2) {
			return (x1 - x2).dot(x1 - x2);
		}
	};
	
	Tensor<double, 2> sample(KDTree::Reader, double scale);
}

#include <fsc/index-inl.h>
