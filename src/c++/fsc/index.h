#pragma once

#include <limits>

#include <kj/one-of.h>

#include "eigen.h"

namespace fsc {
	template<typename Num, int dim>
	struct Box {
		using P = Vec<Num, dim>;
		
		P min_;
		P max_;
		
		Box() = default;
		Box(P min, P max) : min(min), max(max) {}
		
		P center const() {
			return (min + max) / 2;
		}
		
		void expand(P p) {
			for(int i = 0; i < dims; ++i) {
				min[i] = std::min(min[i], p[i]);
				max[i] = std::max(max[i], p[i]);
			}
		}
		
		void expand(Box other) {
			for(int i = 0; i < dims; ++i) {
				min[i] = std::min(min[i], other.min[i]);
				max[i] = std::max(max[i], other.max[i]);
			}
		}
		
		static Box empty() { 
			constexpr Num inf = std::numeric_limits<Num>::infinity();
			return Box(P::Constant(inf), P::Constant(-inf));
		}
	};
	
	template<typename Adapter>
	struct KDPackImpl {
		using Scalar = typename Adapter::Scalar;
		using Leaf   = typename Scalar::Leaf;
		static constexpr int dims = Adapter::dimensions;
		
		using P = Vec<Scalar, dims>;
		
		struct HeapNode {
			P center;
			Box box;
			
			kj::OneOf<Leaf, kj::Vector<HeapNode>> data;	
		}:
		
		using Children = kj::Vector<HeapNode>;
		
		kj::Vector<HeapNode> packOnce(Vector<HeapNode> nodes, size_t leafSize) {
			using kj::Vector;
			
			// Estimate the splitting factor (no. of root nodes per dimension)
			double dfactor = std::pow(((double) nodes.size()) / leafSize, 1.0 / dim);
			size_t factor = (size_t) dfactor;
			
			if(factor < 1)
				factor = 1;
			
			// Compute how we want to subdivide along each dimension for sorting
			Vector<Vector<size_t>> indices(dims + 1);
			
			// First dimension is trivial
			indices[0].resize(2);
			indices[0][0] = 0;
			indices[0][1] = nodes.size();
			
			// For the other dimensions, split up the selected intervals into 'factor'
			// sub-intervals.
			for(auto iDim : kj::range(dims)) {
				const Vector& in = indices[iDim];
				Vector& out = indices[iDim + 1];
				
				const size_t nIn = in.size() - 1;
				const size_t nOut = factor * nIn;
				
				out.resize(nOut + 1);
				out[nOut] = storage.size();
				
				for(auto i : kj::range(nIn)) {
					size_t inStart = in[i];
					size_t inEnd   = in[i+1];
					size_t inCount = inEnd - inStart;
					
					// Every subinterval gets inAll, and remain subintervalls get one more
					size_t inAll = inCount / factor;
					size_t remain = inCount - inAll * factor;
					
					// Assign subintervals (only need to write start points, end points are handled
					// as start points of later intervals
					size_t offset = inStart;
					for(auto iSub : kj::Range(factor)) {
						size_t subSize = iSub < remain ? inAll + 1 : inAll;
						
						out[factor * i + iSub] = offset;
						offset += subSize;
					}
				}
			}
			
			// Execute sorting strategy
			for(auto iDim : kj::range(dims)) {
				const Vector<size_t>& intervals = indices[iDim];
				const size_t nInt = intervals.size() - 1;
				
				auto comparator = [iDim](const HeapNode& n1, const HeapNode& n2) -> bool {
					return n1.center[iDim] < n2.center[iDim];
				}
				
				for(auto iInt : kj::range(nInt)) {
					size_t start = intervals[iInt];
					size_t end   = intervals[iInt + 1];
					
					auto startIt = nodes.begin() + start;
					auto endIt   = nodes.begin() + end;
					
					std::sort(startIt, endIt, comparator);
				}
			}
			
			// Group the nodes inside the intervals into leaves
			const std::vector<size_t> lastStage = indices[dims];
			const size_t nOut = lastStage.size() - 1;
			
			Vector<HeapNode> out(nOut);
			for(auto i : kj::range(nOut)) {
				size_t start = lastStage[i];
				size_t end   = lastState[i + 1];
				size_t n     = end - start;
				
				HeapNode& node = out[i];
				node.box = Box<Scalar, dims>::empty();
				
				Children& children = node.data.init<Children>(n);
				
				for(auto iChild : kj::range(n)) {
					children[n] = mv(nodes[start + iChild]);
					
					const HeapNode& child = children[n];
					node.box.expand(child.box);
				}
				node.center = box.center();
			}
			
			return out;
		}
		
		Vector<HeapNode> pack(Vector<Leaf> leaves, size_t leafSize, Adapter&& adapter) {
			Vector<HeapNode> nodes;
			nodes.reserve(leaves.size());
			
			for(auto& leaf : leaves) {
				HeapNode newNode;
				
				newNode.box = adapter.boundingBox(leaf);
				newNode.center = box.center();
				newNode.data = mv(leaf);
				
				nodes.add(mv(node));
			}
			
			while(nodes.size() > 1) {
				nodes = packOnce(nodes, leafSize);
			}
			
			return nodes;
		}
	};
}