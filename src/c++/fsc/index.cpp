#include <limits>

#include <kj/one-of.h>
#include <kj/function.h>

#include <fsc/index.capnp.h>

#include "eigen.h"

using namespace fsc;
using namespace kj;

using kj::Array;
using kj::Vector;

namespace {
	struct PackNode {
		using Interior = Array<PackNode>;
		using Bound = kj::Tuple<double, double, double>;
		
		kj::Array<Bound> bounds;
		
		Maybe<uint64_t> leaf;
		Array<PackNode> children;
		
		PackNode(const PackNode&) = delete;
		PackNode(PackNode&&) = default;
		
		PackNode(uint64_t leaf, kj::Array<Bound> bounds) :
			bounds(mv(bounds)),
			leaf(leaf)
		{}
		
		PackNode(Interior&& childrenIn) :
			bounds(nullptr),
			children(mv(childrenIn))
		{			
			// Initialize bounds storage
			KJ_REQUIRE(children.size() > 0);
			bounds = kj::heapArray<Bound>(children[0].bounds);
			
			// Initialize bounds to empty boxes
			double inf = std::numeric_limits<double>::infinity();
			for(Bound& b : bounds) {
				get<0>(b) = inf;
				get<1>(b) = -inf;
				get<2>(b) = 0;
			}
			
			// Compute bounds
			for(auto i : kj::indices(bounds)) {
				auto& myMin = get<0>(bounds[i]);
				auto& myMax = get<1>(bounds[i]);
				auto& myCenter = get<2>(bounds[i]);
				
				for(auto& c : children) {
					const auto& childMin = get<0>(c.bounds[i]);
					const auto& childMax = get<1>(c.bounds[i]);
					const auto& childCenter = get<2>(c.bounds[i]);
					
					myMin = std::min(myMin, childMin);
					myMax = std::max(myMax, childMax);
					myCenter += childCenter;
				}
				
				myCenter /= children.size();
			}
		}
		
		int32_t totalCount() const {			
			int32_t result = 1;
			for(const PackNode& child : children) {
				result += child.totalCount();
			}
			
			return result;
		}
	};
		
	kj::Array<PackNode> packStep(kj::Array<PackNode> nodes, size_t desiredLeafSize) {
		KJ_ASSERT(nodes.size() > 0);
		const size_t nDims = nodes[0].bounds.size();
		
		Vector<size_t> indirections;
		for(auto i : kj::indices(indirections))
			indirections[i] = i;
				
		// Packing factor per dimensions
		double dFactor = pow(((double) nodes.size()) / desiredLeafSize, 1.0 / nDims);
		size_t factor = (size_t) factor;
		
		if(factor == 0) factor = 1;
		
		// Compute subdivision indices for all dimensions
		auto indices = kj::heapArray<kj::Vector<size_t>>(nDims + 1);
		
		// First dimension is whole range
		indices[0].resize(2);
		indices[0][0] = 0;
		indices[0][1] = nodes.size();
		
		for(size_t iDim = 1; iDim <= nDims; ++iDim) {
			const auto& in = indices[iDim - 1];
			auto& out = indices[iDim];
			
			out.resize(1 + factor * (in.size() - 1));
			out[out.size() - 1] = nodes.size();
			
			for(size_t i = 0; i < in.size() - 1; ++i) {
				// Subdivide the range from in[i] to in[i+1]
				// info 'factor' sub-ranges and write those into
				// out[factor * i] ... [factor * (i + 1) - 1]
				size_t inStart = in[i];
				size_t inEnd   = in[i+1];
				
				size_t inCount = inEnd - inStart;
				
				// Every subrange gets 'inAll' elements, but 'remain'
				// sub-ranges get one additional element
				size_t inAll = inCount / factor;
				size_t remain = inCount - inAll * factor;
				
				// Compute starting offsets for all the sub-intervals
				size_t offset = inStart;
				for(auto j : kj::range(0, factor)) {
					out[factor * i + j] = offset;
					offset += j < remain ? (inAll + 1) : inAll;
				}
			}
		}
		
		// For all dimensions we need to sort the sub-ranges
		for(auto iDim : kj::range(0, nDims + 1)) {
			auto& ranges = indices[iDim];
			
			auto comparator = [iDim, &nodes](size_t i1, size_t i2) {
				double c1 = get<2>(nodes[i1].bounds[iDim]);
				double c2 = get<2>(nodes[i1].bounds[iDim]);
				
				return c1 < c2;
			};
			
			for(size_t iRange = 0; iRange < ranges.size() - 1; ++iRange) {
				auto itBegin = indirections.begin() + ranges[iRange];
				auto itEnd = indirections.end() + ranges[iRange + 1];
				
				std::sort(itBegin, itEnd, comparator);
			}
		}
		
		const kj::Vector<size_t>& lastStage = indices[nDims];
		const size_t nNodesOut = lastStage.size() - 1;
		
		auto outputBuilder = kj::heapArrayBuilder<PackNode>(nNodesOut);
		for(auto i : kj::range(0, nNodesOut)) {
			size_t start = lastStage[i];
			size_t stop  = lastStage[i + 1];
			
			auto children = kj::heapArrayBuilder<PackNode>(stop - start);
			for(auto iChild : kj::range(start, stop)) {
				size_t targetIndex = indirections[i];
				children.add(mv(nodes[targetIndex]));
			}
			
			outputBuilder.add(PackNode(children.finish()));
		}
		
		return outputBuilder.finish();
	}
	
	kj::Array<PackNode> pack(kj::Array<PackNode> nodes, size_t desiredLeafSize) {
		while(nodes.size() > 1) {
			nodes = packStep(mv(nodes), desiredLeafSize);
		}
		return nodes;
	}
	
	struct KDTreeWriter {
		constexpr static inline size_t MAX_NODES_PER_DIM = (1 << 28) / 3;
		
		kj::Tuple<KDTree::Chunk::Builder, int32_t> findSlot(size_t idx) {
			int32_t csb = out.getChunkSizeBase();
			int32_t rem = out.getChunkRemainder();
			
			if(idx < (csb + 1) * rem) {
				return kj::tuple(out.getChunks()[idx / (csb + 1)], idx % (csb + 1));
			}
			
			idx -= (csb + 1) * rem;
			return kj::tuple(out.getChunks()[rem + idx / csb], idx % csb);
		}
		
		void write(const PackNode& root, KDTree::Builder out) {
			this -> out = out;
			
			size_t nDims = root.bounds.size();
			
			const size_t nNodes = root.totalCount();
			
			// Compute how many chunks we need
			const size_t nChunks = (nNodes * nDims / MAX_NODES_PER_DIM) + 1;
			
			// Compute chunk sizes
			const size_t chunkSizeBase = nNodes / nChunks;
			const size_t remaining = nNodes - (nChunks * chunkSizeBase);
			
			out.setChunkSizeBase(chunkSizeBase);
			out.setChunkRemainder(remaining);
			
			auto chunks = out.initChunks(nChunks);
			for(auto iChunk : kj::indices(chunks)) {
				auto chunk = chunks[iChunk];
				auto bbs = chunk.initBoundingBoxes();
				auto shape = bbs.initShape(3);
				
				size_t chunkSize = iChunk < remaining ? chunkSizeBase + 1 : chunkSizeBase;
				
				bbs.initData(3 * nDims * chunkSize);
				shape.set(0, chunkSize);
				shape.set(1, nDims);
				shape.set(2, 3);
			
				chunk.initNodes(chunkSize);
			}
			
			// Reserve first slot for the root node
			allocOffset = 1;
			
			writeNode(root, 0);
		}
		
		void writeNode(const PackNode& node, size_t slot) {
			auto slotLoc = findSlot(slot);
			KDTree::Chunk::Builder chunk = get<0>(slotLoc);
			
			size_t slotInChunk = get<1>(slotLoc);
			
			// Write bounding box information of node
			auto bbData = chunk.getBoundingBoxes().getData();
			
			for(auto iDim : kj::range(0, nDims)) {
				size_t bbOffset = 3 * (nDims * slotInChunk + iDim);
				
				bbData.set(bbOffset + 0, get<0>(node.bounds[iDim]));
				bbData.set(bbOffset + 1, get<1>(node.bounds[iDim]));
				bbData.set(bbOffset + 2, get<2>(node.bounds[iDim]));
			}
			
			auto outNode = chunk.getNodes()[slotInChunk];
			
			KJ_IF_MAYBE(pLeaf, node.leaf) {
				KJ_ASSERT(node.children.size() == 0);
				outNode.setLeaf(*pLeaf);
			} else {
				// If we have children, we need to allocate some space for them
				auto& children = node.children;
				size_t nChildren = children.size();
				
				size_t childOffset = allocOffset;
				allocOffset += nChildren;
				
				for(auto iChild : kj::indices(children)) {
					writeNode(children[iChild], childOffset + iChild);
				}
				
				auto interior = outNode.initInterior();
				interior.setStart(childOffset);
				interior.setEnd(childOffset + nChildren);
			}
		}
		
	private:
		KDTree::Builder out;
		
		size_t allocOffset = 0;
		size_t nDims = 0;
	};
}