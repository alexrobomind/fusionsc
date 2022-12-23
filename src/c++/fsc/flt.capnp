@0xc196c0f9b77361b4;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("FLT");

using Magnetics = import "magnetics.capnp";
using Geometry = import "geometry.capnp";
using Data = import "data.capnp";
using Index = import "index.capnp";
using Random = import "random.capnp";


# ============================== Service interface ===================================

enum FLTStopReason {
	unknown @0;
	stepLimit @1;
	distanceLimit @2;
	turnLimit @3;
	eventBufferFull @4; # Reserved for internal use. Should not be returned.
	outOfGrid @5;
	nanEncountered @6;
	collisionLimit @7;
	couldNotStep @8;
}

struct FLTRequest {
	# Tensor of shape [3, ...] indicating tracing start points
	startPoints @0 : Data.Float64Tensor;
	field @1 : Magnetics.ComputedField;
	geometry @2 : Geometry.IndexedGeometry;
	
	planes @3 : List(Geometry.Plane);
	
	turnLimit @4 : UInt32;
	distanceLimit @5 : Float64;
	stepLimit @6 : UInt32;
	collisionLimit @7 : UInt32;
	
	stepSize @8 : Float64 = 0.001;
	
	parallelModel : group {
		meanFreePath @9 : Float64;
		meanFreePathGrowth @10 : Float64;
		
		union {
			convectiveVelocity @11 : Float64;
			diffusionCoefficient @12 : Float64;
		}
	}
	
	perpendicularModel : union {
		noDisplacement @13 : Void;
		isotropicDiffusionCoefficient @14 : Float64;
	}
	
	rngSeed @15 : UInt64;
}

struct FieldlineMapping {
	struct MappingFilament {
		# Tensor of shape [N, 3]
		points @0 : Data.Float64Tensor;
		
		# Tensor of shape [N, 3, 3]
		jacobians @1 : Data.Float64Tensor;
	}
	
	struct FilamentPoint {
		filamentIndex @0 : UInt32;
		pointIndex @1 : UInt32;
	}
	
	index @0 : Index.KDTree;
	points @1 : List(FilamentPoint);
	filaments @2 : List(MappingFilament);
}

struct FLTResponse {
	#struct Event {
	#	location @0 : List(Float64);
	#	step @1 : UInt32;
	#	distance @2 : Float64;
	#	turnNo @3 : UInt32;
	#	
	#	lcForward @4 : Float64:
	#	lcBackward @5 : Float64;
	#	
	#	union {
	#		planeIntersection : group {
	#			planeNo @6 : UInt32;
	#			turnNo @7 : UInt32;
	#		}
	#		geometryIntersection : group {
	#			tags @8 : List(Geometry.Tag);
	#			elementIdx @9 : UInt64;
	#		}
	#	}
	#};
	
	# Maximum number of toroidal turns traversed by the fieldline
	nTurns @0 : UInt32;
	
	# Tensor of shape [5] (x, y, z, Lc_fwd, Lc_bwd) + startPoints.shape[1:] + [len(poincarePlanes), nTurns]
	poincareHits @1 : Data.Float64Tensor;
	
	# Tensor of shape [4] (x, y, z, length) + startPoints.shape[1:]
	endPoints @2 : Data.Float64Tensor;
	
	tagNames @3 : List(Text);
	
	# Tensor of shape [len(tagNames)] + startPoints.shape[1:]
	endTags @4 : Data.ShapedList(List(Geometry.TagValue));
	
	# Tensor of shape startPoints.shape[1:]
	stopReasons @5 : Data.ShapedList(List(FLTStopReason));
	
	rngSeed @6 : UInt64;
}

interface FLT {
	trace @0 FLTRequest -> FLTResponse;
}


# ============================== Kernel interface ===================================

# The following structures are internal and not intended to be used in network protocols
# They might change in incompatible versions throughout the library

struct FLTKernelState {
	position @0 : List(Float64);
	numSteps @1 : UInt32;
	distance @2 : Float64;
	turnCount @3 : UInt32;
	phi0 @4 : Float64;
	eventCount @5 : UInt32;
	collisionCount @6 : UInt32;
	
	forward @7 : Bool = true;
	
	nextDisplacementAt @8 : Float64;
	
	displacementCount @9 : UInt32;
	rngState @10 : Random.MT19937State;
}

struct FLTKernelEvent {
	location @0 : List(Float64);
	step @1 : UInt32;
	distance @2 : Float64;

	union {
		notSet @3 : Void;
		phiPlaneIntersection : group {
			planeNo @4 : UInt32;
		}
		newTurn @5 : UInt32;
		geometryHit : group {
			meshIndex @6 : UInt64;
			elementIndex @7 : UInt64;
		}
	}
}

struct FLTKernelData {
	struct Entry {
		stopReason @0 : FLTStopReason;
		state @1 : FLTKernelState;
		events @2 : List(FLTKernelEvent);
	}
	
	data @0 : List(Entry);
}

struct FLTKernelRequest {
	#phiPlanes @0 : List(Float64);
	#
	#turnLimit @1 : UInt32;
	#distanceLimit @2 : Float64;
	#stepLimit @3 : UInt32;
	#
	#stepSize @4 : Float64;
	#
	#grid @5 : Magnetics.ToroidalGrid;
	#
	#collisionLimit @6 : UInt32;
	serviceRequest @0 : FLTRequest;
}