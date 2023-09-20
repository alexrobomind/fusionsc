// DO NOT INCLUDE THIS FILE DIRECTLY
// include index.h instead

#include "kernels/message.h"
#include "intervals.h"

#pragma once

namespace fsc {

template<int dims>
inline CUPNP_FUNCTION typename KDTreeIndex<dims>::FindResult KDTreeIndex<dims>::findNearest(const Vec<double, dims>& x) {
	auto rootNode = tree.getChunks()[0].getNodes()[0];
	
	FindResult result;
	result.distance = std::numeric_limits<double>::infinity();
	
	if(rootNode.hasLeaf()) {
		auto p = closestPoint(x, tree.getChunks()[0].getBoundingBoxes().getData());
		result.distance = distance(x, p);
		result.key = rootNode.getLeaf();
		
		return result;
	}
	
	findNearest(x, result, rootNode);
	return result;
}

template<int dims>
inline CUPNP_FUNCTION void KDTreeIndex<dims>::findNearest(const Vec<double, dims>& x, FindResult& currentClosest, cu::KDTree::Node::Reader node) {
	CUPNP_REQUIRE(!node.hasLeaf());
	
	// Cut down the scan range to the lowest maximum possible distance over all children.
	// There must be a matching leaf within that distance
	auto childStart = node.getInterior().getStart();
	auto childEnd = node.getInterior().getEnd();
	
	for(auto child = childStart; child < childEnd; ++child) {
		auto chunkId = chunkSplit.interval(child);
		auto offset = child - chunkSplit.edge(chunkId);
		
		auto chunk = tree.getChunks()[chunkId];
		
		auto distanceMax = distance(x, furthestPoint(x,
			chunk.getBoundingBoxes().getData().slice(dims * 2 * offset, dims * 2 * (offset + 1))
		));
		
		if(distanceMax < currentClosest.distance)
			currentClosest.distance = distanceMax;
	}
	
	// Limit search to children within that distance
	for(auto child = childStart; child < childEnd; ++child) {
		auto chunkId = chunkSplit.interval(child);
		auto offset = child - chunkSplit.edge(chunkId);
		
		auto chunk = tree.getChunks()[chunkId];
		
		auto distanceMin = distance(x, closestPoint(x,
			chunk.getBoundingBoxes().getData().slice(dims * 2 * offset, dims * 2 * (offset + 1))
		));
		
		if(distanceMin <= currentClosest.distance) {
			auto childNode = chunk.getNodes()[offset];
			if(childNode.hasLeaf()) {
				currentClosest.key = childNode.getLeaf();
				currentClosest.distance = distanceMin;
			} else {
				findNearest(x, currentClosest, childNode);
			}
		}
	}
}

template<int dims>
Vec<double, dims> KDTreeIndex<dims>::closestPoint(Vec<double, dims> x, cupnp::List<double>::Reader bounds) {
	for(int i = 0; i < dims; ++i) {
		double min = bounds[2 * i];
		double max = bounds[2 * i + 1];
		
		if(x[i] < min)
			x[i] = min;
		
		if(x[i] > max)
			x[i] = max;
	}
	
	return x;
}

template<int dims>
Vec<double, dims> KDTreeIndex<dims>::furthestPoint(Vec<double, dims> x, cupnp::List<double>::Reader bounds) {
	for(int i = 0; i < dims; ++i) {
		double min = bounds[2 * i];
		double max = bounds[2 * i + 1];
		
		if(fabs(max - x[i]) > fabs(min - x[i]))
			x[i] = max;
		else
			x[i] = min;
	}
	
	return x;
}

}