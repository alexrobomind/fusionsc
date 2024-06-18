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

interface GeometryLib $Cxx.allowCancellation {
	struct IntersectRequest {
		geometry @0 : IndexedGeometry;
		pStart @1 : Float64Tensor;
		pEnd @2 : Float64Tensor;
	}
	
	struct IntersectResponse {
		struct TagEntry {
			name @0 : Text;
			values @1 : ShapedList(List(TagValue));
		}
		
		position @0 : Float64Tensor;
		lambda @1 : Float64Tensor;
		tags @2 : List(TagEntry);
	}
	
	merge @0 Geometry -> (ref : DataRef(MergedGeometry));
	index @1 (geometry : Geometry, grid : CartesianGrid) -> (indexed : IndexedGeometry);
	planarCut @2 (geometry : Geometry, plane : Plane) -> (edges : Float64Tensor); # edges has shape [3, :, 2]
	
	reduce @3 (geometry : Geometry, maxVertices : UInt32 = 1000000, maxIndices : UInt32 = 1000000) -> (ref : DataRef(MergedGeometry));
	
	weightedSample @4 (geometry : Geometry, scale : Float64 = 0.01) -> (centers : Float64Tensor, areas : List(Float64));
	
	intersect @5 IntersectRequest -> IntersectResponse;
}

struct TagValue {
	union {
		notSet @0 : Void;
		uInt64 @1 : UInt64;
		text   @2 : Text;
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

struct Angle {
	# Angle specified in either degrees or radian.
	
	union {
		deg @0 : Float64;
		rad @1 : Float64;
	}
}

struct Transformed(T) {
	union {
		leaf @0 : T;
		shifted : group {
			shift @1 : List(Float64);
			node  @2 : Transformed(T);
		}
		turned : group {
			angle @3 : Angle;
			center @4 : List(Float64) = [0, 0, 0];
			axis  @5 : List(Float64);
			node  @6 : Transformed(T);
		}
		scaled : group {
			node @7 : Transformed(T);
			scale @8 : List(Float64) = [1, 1, 1];
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

struct GeometryFilter {
	union {
		true @0 : Void;
		and @1 : List(GeometryFilter);
		or @2 : List(GeometryFilter);
		not @3 : GeometryFilter;
	
		isOneOf : group {
			tagName @4 : Text;
			values @5 : List(TagValue);
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
					phiStart @11 : Angle;
					phiEnd @12 : Angle;
					close @16 : Bool = true;
				}
			}
		}
		
		filter : group {
			filter @29 : GeometryFilter;
			geometry @30 : Geometry;
		}
		
		quadMesh : group {
			vertices @31 : DataRef(Float64Tensor);
			wrapU @32 : Bool;
			wrapV @33 : Bool;
		}
		
		# ====== Device-specific ======
		# ----------- W7-X ------------
		
		w7x : union {
			componentsDbMesh     @8 : UInt64;
			componentsDbAssembly @9 : UInt64;
			
			op21HeatShield @23 : Void;
			op21Baffles @24 : Void;
			op21Divertor @25 : Void;
		}
		
		# ----------- J-TEXT ------------
		
		jtext : union {
			target @17 : Void;
			hfsLimiter @18 : Void;
			firstWall @22 : Void;
			
			lfsLimiter @19 : Float64;
			topLimiter @20 : Float64;
			bottomLimiter @21 : Float64;
		}
		
		hsx : group {
			firstWall @26 : Void;
		}
		
		custom : group {
			device @27 : Text;
			name   @28 : Text;
		}
	}
}

#const w7xOp12Divertor   : Geometry = ( w7x = ( componentsDbMeshes = [165, 166, 167, 168, 169] ));
#const w7xOp12Baffles    : Geometry = ( w7x = ( componentsDbMeshes = [320, 321, 322, 323, 324] ));
#const w7xOp12Covers     : Geometry = ( w7x = ( componentsDbMeshes = [325, 326, 327, 328, 329] ));
#const w7xOp12HeatShield : Geometry = ( w7x = ( componentsDbMeshes = [330, 331, 332, 333, 334] ));
#const w7xOp12PumpSlits  : Geometry = ( w7x = ( componentsDbMeshes = [450, 451, 452, 453, 454] ));
#
#const w7xSteelPanels : Geometry =  ( w7x = ( componentsDbAssemblies = [8] ));
#const w7xPlasmaVessel : Geometry = ( w7x = ( componentsDbAssemblies = [9] ));
#
#const w7xOp12Pfcs : Geometry = (
#	combined = [
#		.w7xOp12Divertor, .w7xOp12Baffles, .w7xOp12Covers, .w7xOp12HeatShield,
#		.w7xOp12PumpSlits, .w7xSteelPanels, .w7xPlasmaVessel
#	]
#);

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