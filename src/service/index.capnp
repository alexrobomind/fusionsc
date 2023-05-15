@0xcf8ecbd4f068e2bf;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Index");

using Data = import "data.capnp";

struct KDTree {
	struct Node {
		union {
			leaf @0 : UInt64;
			# Pointer into the list of leaves
			
			interior : group {
				# Range of child nodes
				start @1 : UInt64;
				end @2 : UInt64;
			}
		}
	}
	
	struct Chunk {
		# A chunk of data in the KD-tree.
		# Every chunk holds data for 'chunkSize' tree nodes,
		# where chunkSize = iChunk < chunkRemainder ?
		#   chunkSizeBase + 1 : chunkSizeBase
		
		boundingBoxes @0 : Data.Float64Tensor;
		# A [chunkSize, nDims, 2] array holding, for each node and dimension,
		# minimum extent (0), maximum extent (1) and weighted center estimate (2)
		# along that dimension.
		
		nodes @1 : List(Node);
		# Structural information for nodes (interior pointer or leaf ID)
	}
	
	chunks @0 : List(Chunk);
	
	nTotal @1 : UInt32;
	chunkSize @2 : UInt32;
}

interface KDTreeService {
	struct Chunk {
		boxes @0 : Data.Float64Tensor;
		keys @1 : List(UInt64);
	}
	build @0 (chunks : List(Chunk), leafSize : UInt32 = 1) -> KDTree;
}