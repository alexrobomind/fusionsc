@0xa86c29cb6381b702;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Magnetics");

using D = import "data.capnp";

using G = import "geometry.capnp";

using DataRef = D.DataRef;
using Float64Tensor = D.Float64Tensor;
using ShapedList = D.ShapedList;

# BEGIN [magnetics]

struct FourierSurfaces {
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
	
	rCos @0 : Float64Tensor;
	# Tensor of shape [..., nToroidalCoeffs, nPoloidalCoeffs]
	
	zSin @1 : Float64Tensor;
	# Tensor of shape [..., nToroidalCoeffs, nPoloidalCoeffs]
	
	toroidalSymmetry @2 : UInt32 = 1;
	# Multiplier for toroidal mode number (e.g. 5 for W7-X)
	
	nTurns @3 : UInt32 = 1;
	# Divisor for toroidal mode number. Usually 1, but can equal a
	# different number if the surface has an nTurns * 2pi periodic
	# structure, such as a magnetic island (in this case this would
	# be the island's m number
	
	nTor @4 : UInt32;
	# Maximum toroidal mode number.

	mPol @5 : UInt32;
	# Maximum poloidal mode number
	
	union {
		symmetric @6 : Void;
		# Indicator that the surfaces are Stellarator-symmetric
		
		nonSymmetric : group {
			# Optional Fourier components for surfaces non-symmetric surfaces
			
			rSin @7 : Float64Tensor;
			# Tensor of shape [..., nToroidalCoeffs, nPoloidalCoeffs]
			
			zCos @8 : Float64Tensor;
			# Tensor of shape [..., nToroidalCoeffs, nPoloidalCoeffs]
		}
	}
}

struct ToroidalGrid {
	# Slab-coordinate grid
	# Grid points are:
	#  r = linspace(rMin, rMax, nR)
	#  z = linspace(zMin, zMax, nZ)
	#  phi = linspace(0, 2*pi / nSym, nPhi, endpoint = False)
	
	rMin @0 : Float64;
	rMax @1 : Float64;
	zMin @2 : Float64;
	zMax @3 : Float64;
	
	nSym @4 : UInt32 = 1;
	# Toroidal symmetry of the field (e.g. 5 for W7-X)
	
	nR @5 : UInt64;
	nZ @6 : UInt64;
	nPhi @7 : UInt64;
}

struct ComputedField {
	# Pre-computed (possibly remotely-stored or not yet computed) field
	
	grid @0 : ToroidalGrid;
	# Grid on which the field is (or is being) computed.
	
	data @1 : DataRef(Float64Tensor);
	# Reference to tensor holding the field data.
	#
	# Data shape is [phi, z, r, 3], C (row-major) ordering with last index stride 1
	# When interpreting column-major, data shape is [3, r, z, phi]
	# The field components are ordered Phi, Z, R
	#
	# The data are stored as a reference to avoid repeated upload and download
	# of computed field data between client and server computers. Most use cases
	# pass the data directly from the calculation service to the field line tracer
	# which are on the same machine.
}

# Interface for the resolution of device-specific information

interface FieldResolver {
	resolveField @0 (field : MagneticField, followRefs : Bool = false) -> MagneticField;
	resolveFilament @1 (filament : Filament, followRefs : Bool = false) -> Filament;
}

# Interface for the computation of magnetic fields

interface FieldCalculator $Cxx.allowCancellation {
	struct EvalResult { values @0 : Float64Tensor; }
	compute @0 (field : MagneticField, grid : ToroidalGrid) -> (computedField : ComputedField);
	
	# Interpolate computed field at position. Points must be of shape [3, ...].
	interpolateXyz @1 (field : ComputedField, points : Float64Tensor) -> EvalResult;
	
	# Point-wise field evaluation
	evaluateXyz @2 (field : MagneticField, points : Float64Tensor) -> EvalResult;
	evaluatePhizr @3 (field : MagneticField, points : Float64Tensor) -> EvalResult;
	
	evalFourierSurface @4 (
		surfaces : FourierSurfaces,
		phi : List(Float64),
		theta : List(Float64)
	) -> (
		points : Float64Tensor,
		phiDerivatives : Float64Tensor,
		thetaDerivatives : Float64Tensor
	);
	# Evaluate points on Fourier surfaces
	# Returns tensors of shape [3, ..., nPhi, nTheta]
	# with the remainder being the surfaces shape.
	
	enum RadialModeQuantity {
		field @0;
		flux @1;
	}
	
	calculateRadialModes @5 (
		field : MagneticField, background : MagneticField,
		surfaces : FourierSurfaces,
		nMax : UInt32, mMax : UInt32,
		nPhi : UInt32, nTheta : UInt32,
		nSym : UInt32 = 1,
		useFFT : Bool = true,
		quantity : RadialModeQuantity = field
	) -> (
		cosCoeffs : Float64Tensor, sinCoeffs : Float64Tensor,
		mPol : List(Float64), nTor : List(Float64),
		
		radialValues : Float64Tensor, # Tensor of shape [3, ..., nPhi, nTheta]
		phi : List(Float64), theta : List(Float64),
		
		reCoeffs : Float64Tensor, imCoeffs : Float64Tensor
	);
	# Evaluate points on Fourier surfaces
	#
	# Returns tensors of shape [..., nToroidalCoeffs, nPoloidalCoeffs]
	# - nToroidalCoeffs must be 2 * nTor + 1
	# - nPoloidalCoeffs must be mPol + 1
	#
	# The order of storage for toroidal modes is [0, ..., nTor, -nTor, ..., -1]
	# so that slicing with negative indices can be interpreted
	# as slicing from the end (as is done in NumPy).
	#
	# The order of storage for polidal modes is [0, ..., mPol]
	
	surfaceToMesh @6 (
		surfaces : FourierSurfaces,
		nPhi : UInt32, nTheta : UInt32,
		radialShift : Float64
	) -> (
		merged : DataRef(G.MergedGeometry)
	);
		
	
	# Fourier-mode evaluation
	#calculatePerturbation @4 (
	#	field : MagneticField, surfaces : FourierSurfaces,
	#	nMax : UInt32, mMax : UInt32, toroidalSymmetry : UInt32,
	#	nTor : UInt32, mPol : UInt32
	#) -> (components : Float64Tensor, mPol : Float64Tensor, nTor : Float64Tensor);
}

struct BiotSavartSettings {
	width @0 : Float64;
	stepSize @1 : Float64;
}

struct Filament {
	union {
		# ============= General ============
		
		# Tensor of shape [nPoints, 3]
		inline @0 : Float64Tensor;
		
		ref    @1 : DataRef(Filament);
		nested @2 : Filament;
		
		sum    @5 : List(Filament);
		
		# ========= Device-specific ========
		
		# ------------- W7-X ---------------
		w7x : group {
			# Note: Turn this into union if more variants arrive
			coilsDb @3 : UInt64;
		}
		
		jtext : group {
			# Note: Turn this into union if more variants arrive
			islandCoil @4 : UInt8;
		}
		
		hsx : union {
			mainCoil @6: UInt8;
			auxCoil @7 : UInt8;
		}
		
		custom : group {
			device @8 : Text;
			name   @9 : Text;
		}
	}
}

struct DipoleCloud {
	# Tensor of shape [3, nPoints]
	positions @0 : Float64Tensor;
	
	# Tensor of shape [3, nPoints]
	magneticMoments @1 : Float64Tensor;
	
	# List of length nPoints
	radii @2 : List(Float64);
}

struct AxisymmetricEquilibrium {
	# 2D computational box
	rMin @0 : Float64;
	rMax @1 : Float64;
	zMin @2 : Float64;
	zMax @3 : Float64;
	
	# Poloidal flux function in Weber / rad
	# Shape is [nZ, nR]
	poloidalFlux @4 : Float64Tensor;
	
	# Flux profile range (all fluxes in Weber / rad) for interpolation
	fluxAxis @5 : Float64;
	fluxBoundary @6 : Float64;
	
	# Normalized toroidal field F (F = rMaj * Bt) as a profile of flux
	normalizedToroidalField @7 : List(Float64);
}

# =========================== Device-specifics =============================

# --------------------------------- W7-X -----------------------------------

struct W7XCoilSet {
	union {
		coils : group {
			invertMainCoils @0 : Bool = true;
			biotSavartSettings @1 : BiotSavartSettings = (
				width = 0.01,
				stepSize = 0.01
			);
			
			nWindMain @8 : List(UInt32) = [108, 108, 108, 108, 108, 36, 36];
			nWindTrim @9 : List(UInt32) = [48, 72, 48, 48, 48];
			nWindControl @10 : List(UInt32) = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8];
			
			invertControlCoils @11 : List(Bool) = [
				false, true,
				false, true,
				false, true,
				false, true,
				false, true
			];
			
			union {
				coilsDbSet : group {
					mainCoilOffset    @2 : UInt32 = 160;
					trimCoilIDs       @3 : List(UInt32) = [350, 241, 351, 352, 353];
					controlCoilOffset @4 : UInt32 = 230;
				}
				
				customCoilSet : group {
					mainCoils @5 : List(DataRef(Filament)); # Must have length 70
					trimCoils @6 : List(DataRef(Filament)); # Must have length 5
					controlCoils @7 : List(DataRef(Filament)); # Must have length 10
				}
			}
		}
		
		fields : group {
			mainFields @12 : List(MagneticField);
			trimFields @13 : List(MagneticField);
			controlFields @14 : List(MagneticField);
		}
	}
}

# The following structs describe instruction sets interpreted by the field calculator.
# The nodes in this tree are subdivided into two categories:
#
# - General: These nodes can be interpreted by the field calculator. FieldResolver
#   implementations should aim to only create nodes out of this category. They also
#   don't have to handle these nodes, as they are implicitly handled by FieldResolverBase.
#
# - Device-specific: These nodes contain information specific to one or more devices.
#   These nodes are not understood by the field calculator. The task of field resolvers
#   is to transform these nodes into equivalent representations consisting only of
#   general type nodes. Note that resolvers can fail to resolve nodes e.g. if the
#   required data are not shipped with fscdue to data protection requirements of the
#   individual facilities.

struct MagneticField {	
	# Different ways to calculate the magnetic field
	union {
		# ============================= General =================================
		
		# Calculate the field by summing up contributions
		sum @0 : List(MagneticField);
		
		# Reference to a remotely stored field
		ref @1 : DataRef(MagneticField);
		
		# Magnetic field specified over a grid
		computedField @2 : ComputedField;
		
		# Magnetic field originating from a filament
		filamentField : group {
			current @3 : Float64 = 1;
			biotSavartSettings @4 : BiotSavartSettings;
			filament @5 : Filament;
			windingNo @6 : UInt32 = 1;
		}
		
		scaleBy : group {
			field @7 : MagneticField;
			factor @8 : Float64;
		}
		
		invert @9 : MagneticField;
		nested @10 : MagneticField;
		
		cached : group {
			nested @11 : MagneticField;
			computed @12 : ComputedField;
		}
		
		axisymmetricEquilibrium @20 : AxisymmetricEquilibrium;
		dipoleCloud @21 : DipoleCloud;
		
		transformed @24 : G.Transformed(MagneticField);
		
		# ========================= Device-specific ==========================
		
		# ------------------------------ W7-X --------------------------------

		w7x : union {
			configurationDb : group {
				# The W7-X config specification does not usually include a coil width
				# Therefore, we have to add this information here
				biotSavartSettings @13 : BiotSavartSettings = (
					width = 0.01,
					stepSize = 0.01
				);
				configId  @14 : UInt64;
			}
			
			coilsAndCurrents : group {
				# Currents in the non-planar coils
				nonplanar @ 15 : List(Float64) = [0, 0, 0, 0, 0];
				
				# A list of planar coil currents
				planar    @ 16 : List(Float64) = [0, 0];
				
				# A list of trim coil currents
				trim      @ 17 : List(Float64) = [0, 0, 0, 0, 0];
				
				# A list of control coil currents of either
				# - length 2: Upper and Lower coils
				# - length 5: Coils in each module
				# - length 10: All 10 control coils
				control @18 : List(Float64) = [0, 0];
				
				# The coil set to use. Usually the default theory coils
				coils @19 : W7XCoilSet;
			}
		}
		
		custom : group {
			device @22 : Text;
			name   @23 : Text;
		}
	}
}

# END [magnetics]

const w7xEIMplus252 : MagneticField = (
	w7x = (),
);

# THE FOLLOWING STRUCTURED ARE INTERNAL USE ONLY

struct FourierKernelData {
	field @0 : ComputedField;
	surfaces @1 : FourierSurfaces;
	
	# Data of shape [nSurfaces, 2 * nTor + 1, mPol, 3]
	# holding point samples in phi (dim 1) and theta (dim 2) dimension
	# for Br, Bphi, and Btheta
	pointData @2 : List(Float64);
}