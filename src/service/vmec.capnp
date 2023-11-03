@0xf4c74edb2bac4767;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Vmec");

using Magnetics = import "magnetics.capnp";
using DataPkg = import "data.capnp";

using FTensor = DataPkg.Float64Tensor;
using DataRef = DataPkg.DataRef;

struct VmecSurfaces {
	# Surface Fourier coefficients coefficients for input and output
	# All tensors must have identical shapes
	# - nToroidalCoeffs must be 2 * nTor + 1
	# - nPoloidalCoeffs must be mPol + 1
	#
	# The order of storage for toroidal modes is [0, ..., nTor, -nTor, ..., -1]
	# so that slicing with negative indices can be interpreted
	# as slicing from the end (as is done in NumPy).
	#
	# The order of storage for polidal modes is [0, ..., mPol]
	
	# Tensor of shape [nSurfaces, nToroidalCoeffs, nPoloidalCoeffs]
	rCos @0 : FTensor;
	
	# Tensor of shape [nSurfaces, nToroidalCoeffs, nPoloidalCoeffs]
	zSin @1 : FTensor;
	
	period @2 : UInt32 = 1;
	nTor @3 : UInt32;
	mPol @4 : UInt32;
	
	union {
		symmetric @5 : Void;
		nonSymmetric : group {
			# Tensor of shape [nSurfaces, nToroidalCoeffs, nPoloidalCoeffs]
			rSin @6 : FTensor;
			
			# Tensor of shape [nSurfaces, nToroidalCoeffs, nPoloidalCoeffs]
			zCos @7 : FTensor;
		}
	}
}

struct VmecProfile {
	enum SplineType {
		akima @0;
		cubic @1;
	}
	
	union {
		powerSeries @0 : List(Float64);
		spline : group {
			locations @1 : List(Float64);
			values @2 : List(Float64);
			type @3 : SplineType;
		}
		
		# Profile of the form base * (1 - s ** inner) ** outer
		twoPower : group {
			base @4 : Float64;
			inner @5 : Float64;
			outer @6 : Float64;
		}
	}
}

struct VmecRequest {
	# Run-type specific information
	union {
		fixedBoundary @0 : Void;
		freeBoundary : group {
			# Computed magnetic field
			vacuumField @1 : Magnetics.ComputedField;
			
			# High-level description of magnetic field
			vacuumFieldHl @2 : Magnetics.MagneticField;
		}
	}
	
	startingPoint @3 : VmecSurfaces;
	
	gamma @4 : Float64 = 0;
	massProfile @5 : VmecProfile;
	
	iota : union {
		iotaProfile @6 : VmecProfile;
		fromCurrent : group {
			totalCurrent @7 : Float64;
			union {
				currentProfile @8 : VmecProfile;
				currentDensityProfile @9 : VmecProfile;
			}
		}
	}
	
	runParams : group {
		# Toroidal grid size (NZETA)
		# Default for free boundary: Vacuum field grid size
		# Default for fixed boundary: 2 * nTor + 4
		nGridToroidal @10 : UInt32 = 0;
		
		# Poloidal grid size (NTHETA)
		# Must be at least 2 * mPol + 6
		# Default is 2 * mPol + 6
		nGridPoloidal @11 : UInt32 = 0;
		
		nGridRadial @12 : List(UInt32) = [4, 9, 28, 99];
		maxIterationsPerSequence @13 : UInt32 = 6000;
		convergenceSteps @14 : UInt32 = 100;
		vacuumCalcSteps @15 : UInt32 = 6;
		timeStep @16 : Float32 =  0.9;
		forceToleranceLevels @17 : List(Float32) = [1.0E-5, 1.0E-7, 1.E-10, 1.E-15];
	}
	
	phiEdge @18 : Float64 = 0;
	mPol @19 : UInt32 = 0;
	nTor @20 : UInt32 = 0;
}

struct VmecResult {
	woutNc @0 : DataRef(Data);
	
	surfaces @1 : DataRef(VmecSurfaces);
	volume @2 : Float64;
	energy @3 : Float64;
}

struct VmecResponse {
	# Input file that was generated
	inputFile @0 : Text;
	
	# Output from the code
	stdout @1 : Text;
	stderr @2 : Text;
	
	# Reference to the result
	result : union {
		failed @3 : Text;
		ok @4 : DataRef(VmecResult);
	}
}

interface VmecDriver {	
	run @0 VmecRequest -> VmecResponse;
	computePhiEdge @1 (field : Magnetics.ComputedField, surface : VmecSurfaces) -> (phiEdge : Float64);
}

