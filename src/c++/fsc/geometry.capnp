@0xf89d847c612e7cb3;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Data = import "data.capnp";

using DataRef = Data.DataRef;
using Float64Tensor = Data.Float64Tensor;
using ShapedList = Data.ShapedList;

interface GeometryResolver {
	resolve @0 (geometry : Geometry, followRefs : Bool = false) -> Geometry;
}

interface GeometryLib {
	merge @0 Geometry -> (ref : DataRef(MergedGeometry));
	index @1 (geoRef : DataRef(MergedGeometry), grid : CartesianGrid) -> (ref : DataRef(IndexedGeometry));
}

struct TagValue {
	union {
		notSet @0 : Void;
		uInt64 @1 : UInt64;
	}
}

struct Tag {
	name @0 : Text;
	value @1 : TagValue;
}

struct CartesianGrid {
	xMin @0 : Float64;
	xMax @1 : Float64;
	
	yMin @2 : Float64;
	yMax @3 : Float64;
	
	zMin @4 : Float64;
	zMax @5 : Float64;
	
	nX @6 : UInt64;
	nY @7 : UInt64;
	nZ @8 : UInt64;
}

struct Transformed(T) {
	union {
		leaf @0 : T;
		shifted : group {
			shift @1 : List(Float64);
			node  @2 : Transformed(T);
		}
		turned : group {
			angle @3 : Float64;
			center @4 : List(Float64);
			axis  @5 : List(Float64);
			node  @6 : Transformed(T);
		}
	}
}

struct Geometry {
	tags @0 : List(Tag);
	
	union {
		# ========== General ==========
		combined @1 : List(Geometry);
		
		transformed @2 : Transformed(Geometry);
		
		ref @3 : DataRef(Geometry);
		
		mesh @4 : DataRef(Mesh);
		
		# ====== Device-specific ======
		# ----------- W7-X ------------
		
		componentsDBMeshes     @5 : List(UInt64);
		componentsDBAssemblies @6 : List(UInt64);
	}
}

const w7xOp12Divertor : Geometry = ( componentsDBMeshes = [165, 166, 167, 168, 169] );
const w7xOp12Baffles : Geometry = ( componentsDBMeshes = [320, 321, 322, 323, 324] );
const w7xOp12Covers : Geometry = ( componentsDBMeshes = [325, 326, 327, 328, 329] );
const w7xOp12HeatShield : Geometry = ( componentsDBMeshes = [330, 331, 332, 333, 334] );
const w7xOp12PumpSlits : Geometry = ( componentsDBMeshes = [450, 451, 452, 453, 454] );

const w7xSteelPanels : Geometry = ( componentsDBAssemblies = [8] );
const w7xPlasmaVessel : Geometry = ( componentsDBAssemblies = [9] );

const w7xOp12Pfcs : Geometry = (
	combined = [
		.w7xOp12Divertor, .w7xOp12Baffles, .w7xOp12Covers, .w7xOp12HeatShield,
		.w7xOp12PumpSlits, .w7xSteelPanels, .w7xPlasmaVessel
	]
);

struct Mesh {
	vertices @0 : Float64Tensor;
	# 2D buffer of vertices. First dimension is vertex count, second is dimension.
	
	indices @1 : List(UInt32);
	# Consecutive list of indices into the vertex buffer making up the polygons
	
	# Description on how to interpret the index buffer
	union {
		polyMesh @2 : List(UInt32);
		# A buffer of length n_polys+1, where the i-th polygon spans the range [ polyMesh[i], polyMesh[i+1] [ of the index buffer.
		
		triMesh @3 : Void;
		# Equivalent to polyMesh = [0, 3, 6, 9, ..., indices.size()]. Only valid if indices has a length which is multiple of 3
	}
}

struct MergedGeometry {
	tagNames @0 : List(Text);
	entries  @1 : List(Entry);
	
	struct Entry {
		tags @0 : List(TagValue);
		mesh @1 : Mesh;
	}
}

struct IndexedGeometry {
	struct ElementRef {
		meshIndex @0 : UInt64;
		elementIndex @1 : UInt64;
	}
	
	struct GridEntry {
		elements @0 : List(ElementRef);
	}
	
	base @0 : DataRef(MergedGeometry);
	grid @1 : CartesianGrid;
	data @2 : ShapedList(List(List(ElementRef))); # Note: StructTensor must take the underlying list type, not the element type
}