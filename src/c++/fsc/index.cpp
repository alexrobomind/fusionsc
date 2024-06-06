#include <limits>

#include <kj/one-of.h>
#include <kj/function.h>

#include <fsc/index.capnp.h>

#include "eigen.h"
#include "intervals.h"

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
		if(desiredLeafSize < 2)
			desiredLeafSize = 2;
		
		KJ_ASSERT(nodes.size() > 0);
		const size_t nDims = nodes[0].bounds.size();
		
		Vector<size_t> indirections(nodes.size());
		for(auto i : kj::indices(nodes))
			indirections.add(i);
				
		// Packing factor per dimensions
		double dFactor = pow(((double) nodes.size()) / desiredLeafSize, 1.0 / nDims);
		// KJ_DBG(dFactor);
		size_t factor = (size_t) dFactor;
		
		if(factor == 0) factor = 1;
		
		// Compute subdivision indices for all dimensions
		auto indices = kj::heapArray<kj::Vector<size_t>>(nDims + 1);
		
		// KJ_DBG("Setting up dim 0");
		
		// First dimension is whole range
		indices[0].resize(2);
		indices[0][0] = 0;
		indices[0][1] = nodes.size();
		
		for(size_t iDim = 1; iDim <= nDims; ++iDim) {
			// KJ_DBG("Setting up dim", iDim, factor);
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
		
		/*KJ_DBG("Dimensions computed");
		for(auto i : kj::indices(indices))
			KJ_DBG(i, indices[i]);
		
		KJ_DBG(indirections);*/
		
		// For all dimensions we need to sort the sub-ranges
		for(auto iDim : kj::range(0, nDims)) {
			// KJ_DBG(iDim);
			auto& ranges = indices[iDim];
			
			auto comparator = [iDim, &nodes](size_t i1, size_t i2) {
				double c1 = get<2>(nodes[i1].bounds[iDim]);
				double c2 = get<2>(nodes[i2].bounds[iDim]);
				
				return c1 < c2;
			};
			
			for(auto iRange : kj::range(0, ranges.size() - 1)) {
				// KJ_DBG(iRange, ranges[iRange], ranges[iRange + 1], indirections.size());
				auto itBegin = indirections.begin() + ranges[iRange];
				auto itEnd = indirections.begin() + ranges[iRange + 1];
				
				std::sort(itBegin, itEnd, comparator);
			}
		}
		
		// KJ_DBG("Sort complete", indirections);
		
		const kj::Vector<size_t>& lastStage = indices[nDims];
		const size_t nNodesOut = lastStage.size() - 1;
		
		auto outputBuilder = kj::heapArrayBuilder<PackNode>(nNodesOut);
		for(auto i : kj::range(0, nNodesOut)) {
			size_t start = lastStage[i];
			size_t stop  = lastStage[i + 1];
			
			// KJ_DBG(i, start, stop);
			
			auto children = kj::heapArrayBuilder<PackNode>(stop - start);
			for(auto iChild : kj::range(start, stop)) {
				size_t targetIndex = indirections[iChild];
				// KJ_DBG(iChild, targetIndex);
				children.add(mv(nodes[targetIndex]));
			}
			
			outputBuilder.add(PackNode(children.finish()));
		}
		
		return outputBuilder.finish();
	}
	
	PackNode pack(kj::Array<PackNode> nodes, size_t desiredLeafSize) {
		while(nodes.size() > 1) {
			// KJ_DBG("Beginning pack step", nodes.size());
			nodes = packStep(mv(nodes), desiredLeafSize);
		}
		// KJ_DBG("Pack finished");
		return mv(nodes[0]);
	}
	
	struct KDTreeWriter {
		constexpr static inline size_t MAX_NODES_PER_DIM = (1 << 28) / 3;
		
		KDTreeWriter() :
			out(nullptr),
			split(1, 1)
		{}
		
		void write(const PackNode& root, KDTree::Builder out) {
			this -> out = out;
			
			nDims = root.bounds.size();
			
			const size_t nNodes = root.totalCount();
			const size_t maxChunkSize = MAX_NODES_PER_DIM / nDims;
			split = UnbalancedIntervalSplit(nNodes, maxChunkSize);
						
			// Compute chunk sizes
			
			out.setNTotal(nNodes);
			out.setChunkSize(maxChunkSize);
			
			auto chunks = out.initChunks(split.blockCount());
			for(auto iChunk : kj::indices(chunks)) {
				auto chunk = chunks[iChunk];
				auto bbs = chunk.initBoundingBoxes();
				auto shape = bbs.initShape(3);
				
				size_t chunkSize = split.edge(iChunk + 1) - split.edge(iChunk);
				
				bbs.initData(2 * nDims * chunkSize);
				shape.set(0, chunkSize);
				shape.set(1, nDims);
				shape.set(2, 2);
			
				chunk.initNodes(chunkSize);
			}
			
			// Reserve first slot for the root node
			allocOffset = 1;
			
			writeNode(root, 0);
		}
	
	private:				
		void writeNode(const PackNode& node, size_t slot) {
			int32_t chunkNo = (int32_t) split.interval(slot);
			int32_t offset = slot - split.edge(chunkNo);
			
			KDTree::Chunk::Builder chunk = out.getChunks()[chunkNo];
			
			// Write bounding box information of node
			auto bbData = chunk.getBoundingBoxes().getData();
			
			for(auto iDim : kj::range(0, nDims)) {
				size_t bbOffset = 2 * (nDims * offset + iDim);
				
				bbData.set(bbOffset + 0, get<0>(node.bounds[iDim]));
				bbData.set(bbOffset + 1, get<1>(node.bounds[iDim]));
				// bbData.set(bbOffset + 2, get<2>(node.bounds[iDim]));
			}
			
			auto outNode = chunk.getNodes()[offset];
			
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
		
		KDTree::Builder out;
		UnbalancedIntervalSplit split;
		
		size_t allocOffset = 0;
		size_t nDims = 0;
	};
	
	struct KDServiceImpl : public KDTreeService::Server {
		Promise<void> build(BuildContext ctx) {
			kj::Vector<PackNode> nodes;
			
			size_t nDims = 0;
			bool first = true;
			
			for(auto chunk : ctx.getParams().getChunks()) {
				auto tensor = chunk.getBoxes();
				auto keys = chunk.getKeys();
				
				auto shape = tensor.getShape();
				
				KJ_REQUIRE(shape.size() == 3 || shape.size() == 2, shape.size());
				bool pointMode = shape.size() == 2;
				
				size_t n = shape[0];
				size_t tNDims = shape[1];
				
				if(!pointMode) {
					KJ_REQUIRE(shape[2] == 3);
				}
				
				if(first) {
					nDims = tNDims;
				} else {
					KJ_REQUIRE(tNDims == nDims);
				}
				
				first = false;
				
				auto data = tensor.getData();
				KJ_REQUIRE(data.size() == pointMode ? nDims * n : 3 * nDims * n);
				KJ_REQUIRE(keys.size() == n);
				
				for(auto i : kj::range(0, n)) {
					auto bounds = kj::heapArrayBuilder<PackNode::Bound>(nDims);
					
					for(auto iDim : kj::range(0, nDims)) {
						size_t offset = i * nDims + iDim;
						
						if(pointMode) {
							bounds.add(tuple(
								data[offset], data[offset], data[offset]
							));
						} else {
							offset *= 3;
							bounds.add(tuple(
								data[offset], data[offset + 1], data[offset + 2]
							));
						}
					}
					
					nodes.add(keys[i], bounds.finish());
				}
			}
			
			PackNode packedData = pack(nodes.releaseAsArray(), ctx.getParams().getLeafSize());
			
			KDTreeWriter().write(packedData, ctx.initResults());
			// KJ_DBG(ctx.getResults());
			
			return READY_NOW;
		}
	};
}

namespace fsc {
	Own<KDTreeService::Server> newKDTreeService() {
		return kj::heap<KDServiceImpl>();
	}
}
