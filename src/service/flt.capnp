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

struct FieldlineMapping {
	struct MappingFilament {
		data @0 : List(Float64);
		# Numerical data of the mapping filament.
		# Contains, per phi plane, six numbers.
		# - Numbers 1 and 2 are the r and z coordinates of the
		#   associated mapping point.
		# - Numbers 3 to 6 are, in column-major order, a transformation
		#   matrix from a local UV coordinate system to the RZ coordinates.
		#   which tracks the orientation and shear of the magnetic planes.
		
		# Phi grid information
		phiStart @1 : Float64;
		phiEnd @2 : Float64;
		nIntervals @3 : UInt64;
	}
	
	struct Direction {
		filaments @0 : List(MappingFilament);
		index @1 : Index.KDTree;
	}
	
	fwd @0 : Direction;
	bwd @1 : Direction;
}

struct ReversibleFieldLineMapping {
	struct Section {
		# Tensors of identical shape [nPhi, nZMap, nRMap] which contain R, Z,
		# and forward length values of mapping points.
		r @0 : Data.Float64Tensor;
		z @1 : Data.Float64Tensor;
		traceLen @2 : Data.Float64Tensor;
		
		phiStart @3 : Float64;
		phiEnd @4 : Float64;
	}
	
	# List of phi angles (in radians) corresponding to mapping surfaces
	surfaces @0 : List(Float64);
	
	# List of mapping sections, each covering surface[i] to surface[(i + 1) % surfaces.size()]
	sections @1 : List(Section);
		
	# How many phi sections at start and end point past the end of the phi range
	# axis. These can be used for higher-order interpolation methods to estimate
	# correct derivative values.
	nPad @2 : UInt64;
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
	
	mapping @16 : Data.DataRef(FieldlineMapping);
	forward @17 : Bool = true;
	
	recordEvery @18 : UInt32;
	
	fieldLineAnalysis : union {
		noTask @19 : Void;
		calculateIota : group {
			unwrapEvery @20 : UInt32;
			rAxis @21 : List(Float64);
			zAxis @22 : List(Float64);
		}
		calculateFourierModes : group {
			rAxis @23 : List(Float64);
			zAxis @24 : List(Float64);
			
			# Tensor of shape startPointShape[1:]
			iota @25 : Data.Float64Tensor;
			
			# Maximum n number to calculate
			nMax @26 : UInt32 = 1;
			
			# Maximum m number to calculate
			mMax @27 : UInt32 = 0;
		}
	}
	
	forwardDirection : union {
		field @28 : Void;
		ccw @29 : Void;
	}
}


struct FLTResponse {	
	# Maximum number of toroidal turns traversed by the fieldline
	nTurns @0 : UInt32;
	
	# Tensor of shape [5] (x, y, z, Lc_fwd, Lc_bwd) + [len(poincarePlanes)] + startPoints.shape[1:] + [nTurns]
	poincareHits @1 : Data.Float64Tensor;
	
	# Tensor of shape [4] (x, y, z, length) + startPoints.shape[1:]
	endPoints @2 : Data.Float64Tensor;
	
	tagNames @3 : List(Text);
	
	# Tensor of shape [len(tagNames)] + startPoints.shape[1:]
	endTags @4 : Data.ShapedList(List(Geometry.TagValue));
	
	# Tensor of shape startPoints.shape[1:]
	stopReasons @5 : Data.ShapedList(List(FLTStopReason));
	
	# Tensor of shape startPoints.shape + [max. field line length]
	fieldLines @7 : Data.Float64Tensor;
	
	# Tensor of shape startPoints.shape[1:] + [max. field line length]
	fieldStrengths @8 : Data.Float64Tensor;
	
	rngSeed @6 : UInt64;
}

struct FindAxisRequest {
	startPoint @0 : List(Float64);
	field @1 : Magnetics.ComputedField;
	stepSize @2 : Float64 = 0.001;
	nTurns @3 : UInt32 = 10;
	nIterations @4 : UInt32 = 10;
	nPhi @5 : UInt64 = 20;
}

struct FindLcfsRequest {
	p1 @0 : List(Float64);
	p2 @1 : List(Float64);
	
	tolerance @2 : Float64 = 0.001;
	nScan @3 : UInt32 = 8;
	
	distanceLimit @4 : Float64;
	stepSize @5 : Float64 = 0.001;
	
	field @6 : Magnetics.ComputedField;
	geometry @7 : Geometry.IndexedGeometry;
}

interface FLT $Cxx.allowCancellation {
	trace @0 FLTRequest -> FLTResponse;
	findAxis @1 FindAxisRequest -> (pos : List(Float64), axis : Data.Float64Tensor, meanField : Float64);
	findLcfs @2 FindLcfsRequest -> (pos : List(Float64));
}

struct MappingRequest {
	startPoints @0 : Data.Float64Tensor;
	field @1 : Magnetics.ComputedField;
	
	nPhi @2 : UInt64;
	
	filamentLength @3 : Float64 = 5;
	cutoff @4 : Float64 = 1;
	stepSize @5 : Float64 = 0.001;
	
	dx @6 : Float64 = 0.001;
	
	nSym @7 : UInt32 = 1;
	
	batchSize @8 : UInt32 = 1000;
}

struct RFLMRequest {
	mappingPlanes @0 : List(Float64);
	
	gridR @1 : List(Float64);
	gridZ @2 : List(Float64);
	
	numPlanes @3 : UInt32 = 20;
	numPaddingPlanes @4 : UInt32 = 1;
	
	field @5 : Magnetics.ComputedField;
}

interface Mapper {
	computeMapping @0 MappingRequest -> (mapping : Data.DataRef(FieldlineMapping));
	computeRFLM @1 RFLMRequest -> (mapping : Data.DataRef(ReversibleFieldLineMapping));
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
	
	theta @11 : Float64;
	iota @12 : Float64;
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
		record : group {
			fieldStrength @8 : Float64;
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