@0xc196c0f9b77361b4;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Magnetics = import "magnetics.capnp";
using Data = import "data.capnp";


# ============================== Service interface ===================================

struct FLTRequest {
	# Tensor of shape [3, ...] indicating tracing start points
	startPoints @0 : Data.Float64Tensor;
	field @1 : Magnetics.ComputedField;
	
	poincarePlanes @2 : List(Float64);
	
	turnLimit @3 : UInt32;
	distanceLimit @4 : Float64;
	stepLimit @5 : UInt32;
	
	stepSize @6 : Float64 = 0.001;
}

struct FLTResponse {
	# Maximum number of toroidal turns traversed by the fieldline
	nTurns @0 : UInt32;
	
	# Tensor of shape [3] + startPoints.shape[1:] + [len(poincarePlanes), nTurns]
	poincareHits @1 : Data.Float32Tensor;
}

interface FLT {
	trace @0 FLTRequest -> FLTResponse;
}


# ============================== Kernel interface ===================================

# The following structures are internal and not intended to be used in network protocols
# They might change in incompatible versions throughout the library

enum FLTStopReason {
	unknown @0;
	stepLimit @1;
	distanceLimit @2;
	turnLimit @3;
	eventBufferFull @4;
	outOfGrid @5;
	nanEncountered @6;
}

struct FLTKernelState {
	position @0 : List(Float64);
	numSteps @1 : UInt32;
	distance @2 : Float64;
	turnCount @3 : UInt32;
	phi0 @4 : Float64;
	eventCount @5 : UInt32;
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
	phiPlanes @0 : List(Float64);
	
	turnLimit @1 : UInt32;
	distanceLimit @2 : Float64;
	stepLimit @3 : UInt32;
	
	stepSize @4 : Float64;
	
	grid @5 : Magnetics.ToroidalGrid;
}