@0xcf8ecbd4f068e2bf;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

struct TreeIndexNode(Data, Leaf) {
	data @0 : Data;
	union {
		leaf @1 : Leaf;
		children @2 : List(KDNode(Leaf));
	}
};

interface TreeBuilder {
	
 