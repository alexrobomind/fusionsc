@0xc4b38b12cd771bac;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::devices::w7x");

using Data = import "../data.capnp";
using DataRef = Data.DataRef;

using Magnetics = import "../magnetics.capnp";
using MagneticField = Magnetics.MagneticField;
using Filament = Magnetics.Filament;

interface CoilsDB {
	getCoil @0 (id : UInt64) -> (filament : Filament);
	getConfig @1 (id : UInt64) -> (config : CoilsDBConfig);
}

# Structure that holds instructions on how to compute the field
# for every main coil.
struct CoilFields {
	mainCoils @0 : List(DataRef(MagneticField));
	trimCoils @1 : List(DataRef(MagneticField));
	controlCoils @2 : List(DataRef(MagneticField));
}

# === Client-side JSON interface for CoilsDB ===

struct CoilsDBCoil {
	struct PLFilament {
		x1 @0 : List(Float64);
		x2 @1 : List(Float64);
		x3 @2 : List(Float64);
	}
	polylineFilament @0 : PLFilament;
}

struct CoilsDBCoilInfo {
	struct HistoryEntry {
		timeStamp @0 : UInt64;
		author @1 : Text;
		method @2 : Text;
		comment @3 : Text;
	}
	name @0 : Text;
	machine @1 : Text;
	state @2 : Text;
	quality @3 : Int64;
	author @4 : Text;
	location @5 : Text;
	id @6 : Text;
	method @7 : Text;
	comment @8 : Text;
}

struct CoilsDBConfig {
	coils @0 : List(UInt64);
	currents @1 : List(Float64);
	scale @2 : Float64;
}

interface CoilsDBClient {
	getCoil @0 (id : UInt64) -> (coil : CoilsDBCoil, info : CoilsDBCoilInfo);
	getConfig @1 (id : UInt64) -> (config : CoilsDBConfig);
}