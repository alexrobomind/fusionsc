@0xa86c29cb6381b702;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Data = import "data.capnp";

using DataRef = Data.DataRef;
using Float64Tensor = Data.Float64Tensor;

struct ToroidalGrid {
	rMin @0 : Float64;
	rMax @1 : Float64;
	zMin @2 : Float64;
	zMax @3 : Float64;
	nSym @4 : UInt32;
	
	nR @5 : UInt64;
	nZ @6 : UInt64;
	nPhi @7 : UInt64;
	
	stelSym @8 : Bool;
}

struct ComputedField {
	grid @0 : ToroidalGrid;
	data @1 : DataRef(Float64Tensor);
}

struct MagneticField {
	# Different ways to calculate the magnetic field
	union {
		# Calculate the field by summing up contributions
		sum @0 : List(MagneticField);
		
		# Reference to a remotely stored field
		ref @1 : DataRef(MagneticField);
		
		# Magnetic field specified over a grid
		computedField @2 : ComputedField;
		
		# Magnetic field originating from a filament
		filamentField : group {
			current @3 : Float64;
			width @4 : Float64;
			filament @5 : Filament;
			windingNo @6 : UInt32 = 1;
		}
		
		scaleBy : group {
			field @7 : MagneticField;
			factor @8 : Float64;
		}
		
		invert @9 : MagneticField;
		
		# --- device-specific options, might require special infrastructure to resolve ---

		w7xMagneticConfig : group {
			
			union {
				configurationDB : group {
					# The W7-X config specification does not usually include a coil width
					# Therefore, we have to add this information here
					coilWidth @10 : Float32;
					configID  @11 : UInt64;
				}
				
				coilsAndCurrents : group {
					# Currents in the non-planar coils
					nonplanar @ 12 : List(Float64) = (10000, 10000, 10000, 10000, 10000);
					
					# A list of planar coil currents
					planar    @ 13 : List(Float64) = (0, 0);
					
					# A list of trim coil currents
					trim      @ 14 : List(Float64) = (0, 0, 0, 0, 0);
					
					# A list of control coil currents of either
					# - length 2: Upper and Lower coils
					# - length 5: Coils in each module
					# - length 10: All 10 control coils
					control @15 : List(Float64) = (0, 0);
					
					# The coil set to use. Usually the default theory coils
					coils @16 : W7XCoilSet;
				}
			}
		}
	}
}

struct Filament {
	union {
		inline @0 : Float64Tensor;
		ref    @1 : DataRef(Filament);
		
		w7xCoilsDB @2 : UInt64;
	}
}

# Interface for the resolution of device-specific information

interface FieldResolver {
	resolve @0 (field : MagneticField, followRefs : Bool = false) -> (field : MagneticField);
}

# Interface for the computation of magnetic fields

interface FieldComputer {
	compute @0 (field : MagneticField) -> (computedField : ComputedField);
}

interface FieldComputerFactory {
	get @0 (grid : ToroidalGrid) -> (computer : FieldComputer);
}

struct W7XCoilSet {
	invertMainCoils @0 : Bool = true;
	width @1 : Float64;
	
	union {
		coilsDBSet : group {
			mainCoilOffset    @2 : UInt32 = 160;
			trimCoilIDs       @3 : List(UInt32);
			controlCoilOffset @4 : UInt32;
		}
		
		customCoilSet : group {
			mainCoils @5 : List(DataRef(Filament));
			trimCoils @6 : List(DataRef(Filament));
			controlCoils @7 : List(DataRef(Filament));
		}
	}
}

const w7xEIMplus252 : MagneticField = (
	w7xMagneticConfig = (),
);
