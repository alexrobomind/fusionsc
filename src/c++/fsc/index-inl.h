// DO NOT INCLUDE THIS FILE DIRECTLY
// include index.h instead

#pragma once

// ------------------------------------------------ Tree index -------------------------------------------------

namespace fsc {
	namespace internal {
		//! Packing helper for KD trees
		template<typename Adapter>
		struct KDPackImpl {
			using Scalar = typename Adapter::Scalar;
			using LeafNode   = typename Adapter::LeafNode;
			static constexpr int dims = Adapter::dimensions;
			
			using P = Vec<Scalar, dims>;
			
			struct HeapNode {
				P center;
				Box<Scalar, dims> box;
				
				kj::OneOf<LeafNode, kj::Vector<HeapNode>> data;	
			};
			
			using Children = kj::Vector<HeapNode>;
			
			static kj::Vector<HeapNode> packOnce(kj::Vector<HeapNode>&& nodes, size_t LeafNodeSize) {
				using kj::Vector;
				
				// Estimate the splitting factor (no. of root nodes per dimension)
				double dfactor = std::pow(((double) nodes.size()) / leafSize, 1.0 / dims);
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
				for(auto iDim : kj::range(0, dims)) {
					const Vector<size_t>& in = indices[iDim];
					Vector<size_t>& out = indices[iDim + 1];
					
					const size_t nIn = in.size() - 1;
					const size_t nOut = factor * nIn;
					
					out.resize(nOut + 1);
					out[nOut] = nodes.size();
					
					for(auto i : kj::range(0, nIn)) {
						size_t inStart = in[i];
						size_t inEnd   = in[i+1];
						size_t inCount = inEnd - inStart;
						
						// Every subinterval gets inAll, and remain subintervalls get one more
						size_t inAll = inCount / factor;
						size_t remain = inCount - inAll * factor;
						
						// Assign subintervals (only need to write start points, end points are handled
						// as start points of later intervals
						size_t offset = inStart;
						for(auto iSub : kj::range(0, factor)) {
							size_t subSize = iSub < remain ? inAll + 1 : inAll;
							
							out[factor * i + iSub] = offset;
							offset += subSize;
						}
					}
				}
				
				// Execute sorting strategy
				for(auto iDim : kj::range(0, dims)) {
					const Vector<size_t>& intervals = indices[iDim];
					const size_t nInt = intervals.size() - 1;
					
					auto comparator = [iDim](const HeapNode& n1, const HeapNode& n2) -> bool {
						return n1.center[iDim] < n2.center[iDim];
					};
					
					for(auto iInt : kj::range(0, nInt)) {
						size_t start = intervals[iInt];
						size_t end   = intervals[iInt + 1];
						
						auto startIt = nodes.begin() + start;
						auto endIt   = nodes.begin() + end;
						
						std::sort(startIt, endIt, comparator);
					}
				}
				
				// Group the nodes inside the intervals into leaves
				const kj::Vector<size_t>& lastStage = indices[dims];
				const size_t nOut = lastStage.size() - 1;
				
				Vector<HeapNode> out(nOut);
				for(auto i : kj::range(0, nOut)) {
					size_t start = lastStage[i];
					size_t end   = lastStage[i + 1];
					size_t n     = end - start;
					
					HeapNode& node = out[i];
					node.box = Box<Scalar, dims>::empty();
					
					Children& children = node.data.template init<Children>(n);
					
					for(auto iChild : kj::range(0, n)) {
						children[iChild] = mv(nodes[start + iChild]);
						
						const HeapNode& child = children[n];
						node.box.expand(child.box);
					}
					node.center = node.box.center();
				}
				
				return out;
			}
			
			static HeapNode pack(kj::Vector<LeafNode>&& leaves, size_t leafSize, Adapter&& adapter) {
				using kj::Vector;
				
				Vector<HeapNode> nodes;
				nodes.reserve(leaves.size());
				
				for(LeafNode& leaf : leaves) {
					HeapNode newNode;
					
					newNode.box = adapter.boundingBox(leaf);
					newNode.center = newNode.box.center();
					newNode.data = mv(leaf);
					
					nodes.add(mv(newNode));
				}
				
				while(nodes.size() > 1) {
					nodes = packOnce(mv(nodes), leafSize);
				}
				
				return mv(nodes[0]);
			}
		};
		
		template<typename LeafData>
		struct KD3Adapter {
			using Scalar = double;
			static constexpr int dimensions = 3;
			
			using NodeData = Box3D;
			using LeafNode = TreeNode<NodeData, LeafData>::Reader;
			
			Box<double, 3> boundingBox(Leaf leaf) {
				auto box = leaf.getData();
				
				auto boxMin = box.getMin();
				auto boxMax = box.getMax();
				
				Vec3d min(boxMin.getX(), boxMin.getY(), boxMin.getZ());
				Vec3d max(boxMax.getX(), boxMax.getY(), boxMax.getZ());
				
				return Box<double, 3>(min, max);
			}
			
			static void writeBox(const Box<double, 3>& in, Box3D::Builder out) {
				auto boxMin = out.getMin();
				auto boxMax = out.getMax();
				
				boxMin.setX(in.min[0]);
				boxMin.setY(in.min[1]);
				boxMin.setZ(in.min[2]);
				
				boxMax.setX(in.max[0]);
				boxMax.setY(in.max[1]);
				boxMax.setZ(in.max[2]);
			}
		};
		
		template<typename LeafData>
		struct KD2Adapter {
			using Scalar = double;
			static constexpr int dimensions = 2;
			
			using NodeData = Box2D;
			using LeafNode = TreeNode<NodeData, LeafData>::Reader;
			
			Box<double, 2> boundingBox(Leaf leaf) {
				auto box = leaf.getData();
				
				auto boxMin = box.getMin();
				auto boxMax = box.getMax();
				
				Vec2d min(boxMin.getX(), boxMin.getY());
				Vec2d max(boxMax.getX(), boxMax.getY());
				
				return Box<double, 2>(min, max);
			}
			
			static void writeBox(const Box<double, 2>& in, Box2D::Builder out) {
				auto boxMin = out.getMin();
				auto boxMax = out.getMax();
				
				boxMin.setX(in.min[0]);
				boxMin.setY(in.min[1]);
				
				boxMax.setX(in.max[0]);
				boxMax.setY(in.max[1]);
			}
		};
		
		/**
		 * Helper class for packing up a user-defined leaf type.
		 *
		 * \tparam Adapter Packing adapter class. Needs to provide the following information:
		 *   Types:
		 *     - Scalar : Number type to use for packing
		 *     - Leaf   : Type of the leaf nodes to be packed up. Must be move-assignable.
		 *     - NodeData     : Cap'n'proto type corresponding to the bounding box type.
		 *
		 *   Static members:
		 *     - constexpr int dims : Dimensionality of the tree.
		 *
		 *   Methods (not neccessarily static):
		 *     - Box<Scalar, dims> boundingBox(const Leaf&) : Constructs the bounding box of a leaf type
		 *     - void writeBox(const Box<Scalar, dims>&, typename NodeData::Builder) : Writes out a bounding box
		 */
		template<typename Adapter>
		void buildKDTree(typename capnp::List<TreeNode<typename Adapter::NodeData, typename Adapter::LeafData>>::Reader input, size_t leafSize, typename TreeNode<typename Adapter::NodeData, typename Adapter::LeafData>::Builder output, Adapter&& adapter) {
			using kj::Vector;
			
			using Leaf = typename Adapter::Leaf;
			
			using Packer = KDPackImpl<Adapter>;
			using HeapNode = typename Packer::HeapNode;
			
			Vector<Leaf> packInput;
			packInput.reserve(input.size());
			for(auto i : input)
				packInput.add(i);
			
			typename Packer::HeapNode result = Packer::pack(mv(packInput), leafSize, fwd<Adapter>(adapter));

			kj::Function<void(const HeapNode& in, typename TreeNode<typename Adapter::NodeData>::Builder out)> transferNode = [&transferNode](const HeapNode& in, typename TreeNode<typename Adapter::NodeData>::Builder out) {
				Adapter::writeBox(in.box, out.initData());
				
				if(in.data.template is<Leaf>()) {
					out.setLeaf(in.data.template get<Leaf>().getLeaf());
				} else {
					const typename Packer::Children& children = in.data.template get<typename Packer::Children>();
					auto outChildren = out.initChildren(children.size());
					
					for(auto i : kj::range(0, children.size())) {
						transferNode(children[i], outChildren[i]);
					}
				}
			};
			
			transferNode(result, output);
		}
	}
	
	template<typename LeafData>
	buildKDTree2(typename capnp::List<TreeNode<Box2D, LeafData>>::Reader input, size_t leafSize, typename TreeNode<Box2D, LeafData>::Builder output) {
		using Adapter = internal::KD2Adapter<LeafData>;
		internal::buildKDTree(input, leafSize, output, Adapter());
	}
	
	template<typename LeafData>
	buildKDTree3(typename capnp::List<TreeNode<Box2D, LeafData>>::Reader input, size_t leafSize, typename TreeNode<Box2D, LeafData>::Builder output) {
		using Adapter = internal::KD3Adapter<LeafData>;
		internal::buildKDTree(input, leafSize, output, Adapter());
	}
}