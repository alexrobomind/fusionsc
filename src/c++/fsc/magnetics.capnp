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

struct MagneticField {
	# Different ways to calculate the magnetic field
	union {
		# Calculate the field by summing up contributions
		sum @0 : List(MagneticField);
		
		# Reference to a remotely stored field
		ref @1 : DataRef(MagneticField);
		
		# Magnetic field specified over a grid
		gridField : group {
			grid @2 : ToroidalGrid;
			data @3 : Float64Tensor;
		}
		
		# Magnetic field originating from a filament
		filamentField : group {
			current @4 : Float64;
			width @5 : Float64;
			filament @6 : Filament;
			windingNo @7 : UInt32 = 1;
		}
		
		scaleBy : group {
			field @8 : MagneticField;
			factor @9 : Float64;
		}
		
		invert @10 : MagneticField;
		
		# --- device-specific options, might require special infrastructure to resolve ---

		w7xMagneticConfig : group {
			# The W7-X config specification does not usually include a coil width
			# Therefore, we have to add this information here
			coilWidth @11 : Float32;
			
			union {
				configurationDB @12 : UInt64;
				
				coilsAndCurrents : group {
					# Currents in the non-planar coils
					np1 @13 : Float64 = 10000;
					np2 @14 : Float64 = 10000;
					np3 @15 : Float64 = 10000;
					np4 @16 : Float64 = 10000;
					np5 @17 : Float64 = 10000;
					
					# Currents in the planar coils
					pA @18 : Float64 = 0;
					pB @19 : Float64 = 0;
					
					# Currents in the trim coils
					trim1 @20 : Float64 = 0;
					trim2 @21 : Float64 = 0;
					trim3 @22 : Float64 = 0;
					trim4 @23 : Float64 = 0;
					trim5 @24 : Float64 = 0;
					
					# A list of control coils of either
					# - Length 0: No control coils
					# - length 2: Upper and Lower coils
					# - length 5: Coils in each module
					# - length 10: All 10 control coils
					controlCoils @25 : List(Float64);
					
					# The coil set to use. Usually the default theory coils
					coils @26 : W7XCoilSet;
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

struct W7XCoilSet {
	invertMainCoils @0 : Bool = true;
	
	union {
		coilsDBSet : group {
			mainCoilOffset    @1 : UInt32 = 160;
			trimCoilOffset    @2 : UInt32; # TODO: Missing default trim coils
			controlCoilOffset @3 : UInt32; # TODO: Missing default control coils
		}
		
		customCoilSet : group {
			nonplanarCoils @4 : List(Filament);
			planarCoils @5 : List(Filament);
			controlCoils @6 : List(Filament);
		}
	}
}

const w7xEIMplus252 : MagneticField = (
	w7xMagneticConfig = (),
);
