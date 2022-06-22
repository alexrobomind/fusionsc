#pragma once

#include <limits>

#include <kj/one-of.h>

#include <index.capnp.h>

#include "eigen.h"

using namespace fsc

namespace {
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
		using Leaf   = typename Adapter::Leaf;
		static constexpr int dims = Adapter::dimensions;
		
		using P = Vec<Scalar, dims>;
		
		struct HeapNode {
			P center;
			Box box;
			
			kj::OneOf<Leaf, kj::Vector<HeapNode>> data;	
		}:
		
		using Children = kj::Vector<HeapNode>;
		
		static kj::Vector<HeapNode> packOnce(Vector<HeapNode> nodes, size_t leafSize) {
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
		
		static Vector<HeapNode> pack(Vector<Leaf> leaves, size_t leafSize, Adapter&& adapter) {
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
			
			return mv(nodes[0]);
		}
	};
	
	struct Box3DAdapter	{
		using Scalar = double;
		constexpr int dimensions = 3;
		
		using CP = Box3D;
		
		using Leaf = TreeNode<CP>::Reader;
		
		Box<double, 3> boundingBox(Leaf box) {
			auto box = leaf.getData();
			
			auto boxMin = box.getMin();
			auto boxMax = box.getMax();
			
			Vec3d min(boxMin.getX(), boxMin.getY(), boxMin.getZ());
			Vec3d max(boxMax.getX(), boxMax.getY(). boxMax.getZ());
			
			return Box<double, 3>(min, max);
		}
		
		static void writeBox(const Box<double, 3>& in, Box3D::Builder out) {
			auto boxMin = box.getMin();
			auto boxMax = box.getMax();
			
			boxMin.setX(in.min[0]);
			boxMin.setY(in.min[1]);
			boxMin.setZ(in.min[2]);
			
			boxMax.setX(in.max[0]);
			boxMax.setY(in.max[1]);
			boxMax.setZ(in.max[2]);
		}
	};
	
	struct Box2DAdapter	{
		using Scalar = double;
		constexpr int dimensions = 2;
		
		using CP = Box2D;
		using Leaf = TreeNode<CP>::Reader;
		
		Box<double, 2> boundingBox(Leaf leaf) {
			auto box = leaf.getData();
			
			auto boxMin = box.getMin();
			auto boxMax = box.getMax();
			
			Vec3d min(boxMin.getX(), boxMin.getY());
			Vec3d max(boxMax.getX(), boxMax.getY());
			
			return Box<double, 2>(min, max);
		}
		
		static void writeBox(const Box<double, 3>& in, Box3D::Builder out) {
			auto boxMin = box.getMin();
			auto boxMax = box.getMax();
			
			boxMin.setX(in.min[0]);
			boxMin.setY(in.min[1]);
			
			boxMax.setX(in.max[0]);
			boxMax.setY(in.max[1]);
		}
	};
	
	template<typename Adapter>
	buildKDTree(capnp::List<TreeNode<typename Adapter::CP>>::Reader input, size_t leafSize, TreeNode<typename Adapter::CP>::Builder output, Adapter&& adapter) {
		using Leaf = Adapter::Leaf;
		
		using Packer = PackImpl<Adapter>;
		using HeapNode = typename Packer::HeapNode;
		
		Vector<Leaf> packInput(input.begin(), input.end());
		Leaf result = PackImpl<Adapter>::pack(packInput, leafSize, fwd<Adapter>(adapter));

		auto tranferNode = [](const HeapNode& in, TreeNode<typename Adapter::CP>::Builder out) {
			Adapter::writeBox(in.box, out.initBox());
			
			if(in.data.is<Leaf>()) {
				out.setLeaf(in.data.get<Leaf>());
			} else {
				auto& children = in.data.get<typename Packer::Children>();
				auto outChildren = out.initChildren(children.size());
				
				for(auto i : kj::range(children.size())) {
					transferNode(children[i], outChildren[i]);
				}
			}
		};
		
		transferNode(result, output);
	}
	
	struct TreeBuliderImpl : public TreeBuilder::Server {
		
	};
}