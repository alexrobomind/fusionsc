@0xa86c29cb6381b702;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Magnetics");

using Data = import "data.capnp";

using DataRef = Data.DataRef;
using Float64Tensor = Data.Float64Tensor;

# BEGIN [magnetics]

struct ToroidalGrid {
	rMin @0 : Float64;
	rMax @1 : Float64;
	zMin @2 : Float64;
	zMax @3 : Float64;
	nSym @4 : UInt32 = 1;
	
	nR @5 : UInt64;
	nZ @6 : UInt64;
	nPhi @7 : UInt64;
}

# Data shape is [phi, z, r, 3], C (row-major) ordering with last index stride 1
# When interpreting column-major, data shape is [3, r, z, phi]
# The field components are ordered Phi, Z, R
struct ComputedField {
	grid @0 : ToroidalGrid;
	data @1 : DataRef(Float64Tensor);
}

# Interface for the resolution of device-specific information

interface FieldResolver {
	resolveField @0 (field : MagneticField, followRefs : Bool = false) -> MagneticField;
	resolveFilament @1 (filament : Filament, followRefs : Bool = false) -> Filament;
}

# Interface for the computation of magnetic fields

interface FieldCalculator $Cxx.allowCancellation {
	compute @0 (field : MagneticField, grid : ToroidalGrid) -> (computedField : ComputedField);
	evaluateXYZ @1 (field : ComputedField, points : Float64Tensor) -> (values : Float64Tensor);
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
	}
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
	}
}

# END [magnetics]

const w7xEIMplus252 : MagneticField = (
	w7x = (),
);
