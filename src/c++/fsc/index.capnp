@0xcf8ecbd4f068e2bf;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Index");

using Data = import "data.capnp";

# Note: There is currently a bug in MSVC that prevents me from laying out the fields as I want
# inside a parametrized struct. Until that bug is fixed, I will have to do without type
# parameters.

struct TreeNode(Data, Leaf) {
	data @0 : Data;
	
	union {
		leaf @1 : Leaf;
		children @2 : List(TreeNode(Data, Leaf));
	}
}

struct Box2D {
	min : group {
		x @0 : Float64;
		y @1 : Float64;
	}
	max : group {
		x @2 : Float64;
		y @3 : Float64;
	}
}

struct Box3D {
	min : group {
		x @0 : Float64;
		y @1 : Float64;
		z @2 : Float64;
	}
	max : group {
		x @3 : Float64;
		y @4 : Float64;
		z @5 : Float64;
	}
}

struct BoxND {
	min @0 : List(Float64);
	max @1 : List(Float64);
}

interface TreeBuilder {
	buildKDTree2 @0 [Leaf] (nodes : TreeNode(Box2D, Leaf), leafSize : UInt32) -> (ref : Data.DataRef(TreeNode(Box2D, Leaf)));
	buildKDTree3 @1 [Leaf] (nodes : TreeNode(Box3D, Leaf), leafSize : UInt32) -> (ref : Data.DataRef(TreeNode(Box3D, Leaf)));
}