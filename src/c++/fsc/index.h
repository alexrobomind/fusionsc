#pragma once

#include <fsc/index.capnp.h>

namespace fsc {
	template<typename LeafData>
	buildKDTree2(typename capnp::List<TreeNode<Box2D, LeafData>>::Reader input, size_t leafSize, typename TreeNode<Box2D, LeafData>::Builder output);
	
	template<typename LeafData>
	buildKDTree3(typename capnp::List<TreeNode<Box2D, LeafData>>::Reader input, size_t leafSize, typename TreeNode<Box2D, LeafData>::Builder output);
}

#include <fsc/index-inl.h>