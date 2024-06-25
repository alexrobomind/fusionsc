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

struct ReversibleFieldlineMapping {
	struct Section {
		# Tensors of identical shape [nPhi, nZMap, nRMap] which contain R, Z,
		# and forward length values of mapping points.
		r @0 : Data.Float64Tensor;
		z @1 : Data.Float64Tensor;
		traceLen @2 : Data.Float64Tensor;
		
		phiStart @3 : Float64;
		phiEnd @4 : Float64;
		
		u0 @5 : Float64;
		v0 @6 : Float64;
		
		struct Inverse {
			rMin @0 : Float64;
			rMax @1 : Float64;
			zMin @2 : Float64;
			zMax @3 : Float64;
			
			# Tensors of shape [nPhi, nZ, nR] containing U and V values on
			# a slab grid with phi planes matching the section.
			u @4 : Data.Float64Tensor;
			v @5 : Data.Float64Tensor;
		}
		inverse @7 : Inverse;
	}
	
	# List of phi angles (in radians) corresponding to mapping surfaces
	surfaces @0 : List(Float64);
	
	# List of mapping sections, each covering surface[i] to surface[(i + 1) % surfaces.size()]
	sections @1 : List(Section);
		
	# How many phi sections at start and end point past the end of the phi range
	# axis. These can be used for higher-order interpolation methods to estimate
	# correct derivative values.
	nPad @2 : UInt64;
	
	struct State {
		u @0 : Float64;
		v @1 : Float64;
		
		section @2 : UInt64;
		
		phi @3 : Float64;
		lenOffset @4 : Float64;
	}
}

struct FLTRequest {
	struct AdaptiveStepControl {
		min @0 : Float64 = 0;
		max @1 : Float64 = inf;
		targetError @2 : Float64 = 1e-6;
		relativeTolerance @3 : Float64 = 1;
		
		errorUnit : union {
			step  @4 : Void;
			integratedOver @5 : Float64 = 1e4;
		}
	}
	
	# Tensor of shape [3, ...] indicating tracing start points
	startPoints @0 : Data.Float64Tensor;
	field @1 : Magnetics.ComputedField;
	geometry @2 : Geometry.IndexedGeometry;
	
	planes @3 : List(Geometry.Plane);
	
	turnLimit @4 : UInt32;
	distanceLimit @5 : Float64;
	stepLimit @6 : UInt32;
	collisionLimit @7 : UInt32;
	
	# Minimum distance to traverse before
	# collisions will be processed
	ignoreCollisionsBefore @39 : Float64;
	
	stepSize @8 : Float64 = 0.001;
	stepSizeControl : union {
		fixed @31 : Void;
		adaptive @32 : AdaptiveStepControl;
	}
	
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
		rzDiffusionCoefficient @30 : Float64;
	}
	
	rngSeed @15 : UInt64;
	
	mapping @16 : Data.DataRef(ReversibleFieldlineMapping);
	forward @17 : Bool = true;
	
	recordEvery @18 : UInt32;
	
	fieldLineAnalysis : union {
		noTask @19 : Void;
		calculateIota : group {
			unwrapEvery @20 : UInt32;
			rAxis @21 : List(Float64);
			zAxis @22 : List(Float64);
			islandM @37 : UInt32 = 1;
		}
		calculateFourierModes : group {			
			unusedRAxis @23 : List(Float64);
			unusedZAxis @24 : List(Float64);
			
			# Tensor of shape startPointShape[1:]
			iota @25 : Data.Float64Tensor;
			
			# Maximum n number to calculate
			nMax @26 : UInt32 = 1;
			
			# Maximum m number to calculate
			mMax @27 : UInt32 = 0;
			
			# How often to record Fourier points
			recordEvery @33 : UInt32 = 1;
			
			# Multiplier for toroidal mode numbers
			toroidalSymmetry @ 34 : UInt32 = 1;
			
			# If n * iota + m of a mode is within this value
			# of a lower-n mode, the mode will be discarded
			# from reconstruction
			modeAliasingThreshold @35 : Float64 = 0.001;
			
			stellaratorSymmetric @36 : Bool = false;
			
			# Denominator for toroidal mode numbers
			islandM @38 : UInt32 = 1;
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
	
	fieldLineAnalysis : union {
		noTask @10 : Void;
		
		# Tensor of shape startPoints.shape[1:]
		iotas @11 : Data.Float64Tensor;
		
		fourierModes : group {			
			# Surface Fourier coefficients coefficients for input and output
			# All tensors must have identical shapes
			# - nToroidalCoeffs must be 2 * nMax + 1
			# - nPoloidalCoeffs must be mMax + 1
			#
			# The order of storage for toroidal modes is [0, ..., nTor, -nTor, ..., -1]
			# so that slicing with negative indices can be interpreted
			# as slicing from the end (as is done in NumPy).
			#
			# The order of storage for poloidal modes is [0, ..., mPol]
			
			# Surfaces
			surfaces @12 : Magnetics.FourierSurfaces;
			
			# Theta values (poloidal angle) of starting points
			theta0 @13 : Data.Float64Tensor;
			
			# Mode number tensors of shape [nToroidalCoeffs, nPoloidalCoeffs]
			# These are float tensors because we want to support fractional mode numbers
			# for magnetic island support (phi ranging from 0 to e.g. 12 pi)
			mPol @14 : List(Float64);
			nTor @15 : List(Float64);
		}
	}
	
	# Number of steps
	numSteps @9 : Data.UInt64Tensor;
	
	rngSeed @6 : UInt64;
}

struct FindAxisRequest {
	startPoint @0 : List(Float64);
	field @1 : Magnetics.ComputedField;
	stepSize @2 : Float64 = 0.001;
	nTurns @3 : UInt32 = 10;
	nIterations @4 : UInt32 = 10;
	nPhi @5 : UInt64 = 20;
	
	mapping @6 : Data.DataRef(ReversibleFieldlineMapping);
	
	stepSizeControl : union {
		fixed @7 : Void;
		adaptive @8 : FLTRequest.AdaptiveStepControl;
	}
	
	islandM @9 : UInt32 = 1;
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
	
	mapping @8 : Data.DataRef(ReversibleFieldlineMapping);
	
	stepSizeControl : union {
		fixed @9 : Void;
		adaptive @10 : FLTRequest.AdaptiveStepControl;
	}
}

interface FLT $Cxx.allowCancellation {
	trace @0 FLTRequest -> FLTResponse;
	findAxis @1 FindAxisRequest -> (pos : List(Float64), axis : Data.Float64Tensor, meanField : Float64);
	findLcfs @2 FindLcfsRequest -> (pos : List(Float64));
}

struct RFLMRequest {
	mappingPlanes @0 : List(Float64);
	
	gridR @1 : List(Float64);
	gridZ @2 : List(Float64);
	
	numPlanes @3 : UInt32 = 20;
	numPaddingPlanes @4 : UInt32 = 1;
	
	field @5 : Magnetics.ComputedField;
	
	distanceLimit @6 : Float64 = 1e3;
	stepSize @7 : Float64 = 0.001;
	
	u0 @8 : List(Float64) = [0.5];
	v0 @9 : List(Float64) = [0.5];
}

interface Mapper {	
	computeRFLM @0 RFLMRequest -> (mapping : Data.DataRef(ReversibleFieldlineMapping));
}

# ==================================== FLT configuration ============================

struct FLTConfig {
	# Settings regarding the size tuning of the event buffer
	# targetTotalMB is used to compute an estimated buffer size
	# dependent on the number of start points. minSize and maxSize
	# are then used to clamp the computed value.
	eventBuffer : group {
		minSize @0 : UInt32 = 100;
		maxSize @1 : UInt32 = 2500;
		targetTotalMb @2 : UInt64 = 500;
	}
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
	
	phi @11 : Float64;
	theta @12 : Float64;
	
	stepSize @13 : Float64;
}

struct FLTKernelEvent {
	x @0 : Float64;
	y @1 : Float64;
	z @2 : Float64;
	
	step @3 : UInt32;
	distance @4 : Float64;

	union {
		notSet @5 : Void;
		phiPlaneIntersection : group {
			planeNo @6 : UInt32;
		}
		newTurn @7 : UInt32;
		geometryHit : group {
			meshIndex @8 : UInt64;
			elementIndex @9 : UInt64;
		}
		record : group {
			fieldStrength @10 : Float64;
		}
		fourierPoint : group {
			phi @11 : Float64;
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

struct RFLMKernelData {
	states @0 : List(ReversibleFieldlineMapping.State);
	phiValues @1 : List(Float64);
}