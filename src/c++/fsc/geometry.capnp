@0xf89d847c612e7cb3;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Geometry");

using Data = import "data.capnp";

using DataRef = Data.DataRef;
using Float64Tensor = Data.Float64Tensor;
using ShapedList = Data.ShapedList;

interface GeometryResolver {
	resolveGeometry @0 (geometry : Geometry, followRefs : Bool = false) -> Geometry;
}

interface GeometryLib {
	merge @0 Geometry -> (ref : DataRef(MergedGeometry));
	index @1 (geometry : Geometry, grid : CartesianGrid) -> (indexed : IndexedGeometry);
	planarCut @2 (geometry : Geometry, plane : Plane) -> (edges : Float64Tensor); # edges has shape [3, :, 2]
}

struct TagValue {
	union {
		notSet @0 : Void;
		uInt64 @1 : UInt64;
		text   @2 : Text;
		
		#w7xComponent : group {
		#	union {
		#		baffle : group {
		#			module @4 : UInt8;
		#			upper @5 : Bool;
		#			row @6 : Text;
		#			tileNo @7 : UInt8;
		#		}
		#		
		#		heatShieldTile : group {
		#			halfModule @8 : UInt8;
		#			section @9 : Text;
		#			tileNo @10 : UInt8;
		#		}
		#		
		#		divertorPart : group {
		#			module @11 : UInt8;
		#			upper @12 : Bool;
		#			section @13 : Text;
		#			partNo @14 : UInt8;
		#			
		#			type : union {
		#				finger @15 : Void;
		#				fingerProtector @16 : Void;
		#				toroidalClosure @17 : Void;
		#				
		#	}
		#}
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
	
	nX @6 : UInt32;
	nY @7 : UInt32;
	nZ @8 : UInt32;
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
			center @4 : List(Float64) = [0, 0, 0];
			axis  @5 : List(Float64);
			node  @6 : Transformed(T);
		}
	}
}

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

struct Plane {
	orientation : union {
		# Indicates that this plane is a phi-oriented toroidal HALF-plane
		phi @0 : Float64;
		
		# Indicates that this plane is a usual plane normal to the given direction
		normal @1 : List(Float64);
	}
	
	center @2 : List(Float64);
}

struct Geometry {
	tags @0 : List(Tag);
	
	union {
		# ========== General ==========
		combined @1 : List(Geometry);
		
		transformed @2 : Transformed(Geometry);
		
		ref @3 : DataRef(Geometry);
		nested @4 : Geometry;
		
		mesh @5 : DataRef(Mesh);
		
		merged @6 : DataRef(MergedGeometry);
		indexed @7 : IndexedGeometry;
		
		wrapToroidally : group {
			r @13 : List(Float64);
			z @14 : List(Float64);
			nPhi @15 : UInt32;
			union {
				fullTorus @10 : Void;
				phiRange : group {
					phiStart @11 : Float64;
					phiEnd @12 : Float64;
				}
			}
		}
		
		# ====== Device-specific ======
		# ----------- W7-X ------------
		
		componentsDBMeshes     @8 : List(UInt64);
		componentsDBAssemblies @9 : List(UInt64);
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
	
	struct IndexData {
		gridContents @0 : ShapedList(List(List(ElementRef)));
	}
	
	base @0 : DataRef(MergedGeometry);
	grid @1 : CartesianGrid;
	data @2 : DataRef(IndexData);
}