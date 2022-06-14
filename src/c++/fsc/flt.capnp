@0xc196c0f9b77361b4;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Magnetics = import "magnetics.capnp";
using Data = import "data.capnp";

# The following structures are internal and not intended to be used in network protocols
# They might change in incompatible versions throughout the library

enum FLTStopReason {
	unknown @0;
	stepLimit @1;
	distanceLimit @2;
	turnLimit @3;
	eventBufferFull @4;
	outOfGrid @5;
}

struct FLTKernelState {
	position @0 : List(Float32);
	numSteps @1 : UInt32;
	distance @2 : Float32;
	turnCount @3 : UInt32;
	phi0 @4 : Float32;
	eventCount @5 : UInt32;
}

struct FLTKernelEvent {
	location @0 : List(Float32);
	step @1 : UInt32;
	distance @2 : Float32;

	union {
		outOfGrid @3 : Void;
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
	phiPlanes @0 : List(Float32);
	
	turnLimit @1 : UInt32;
	distanceLimit @2 : Float32;
	stepLimit @3 : UInt32;
	
	stepSize @4 : Float32;
	
	grid @5 : Magnetics.ToroidalGrid;
}