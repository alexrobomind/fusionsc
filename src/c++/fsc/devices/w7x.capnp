@0xc4b38b12cd771bac;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::devices::w7x");

using Data = import "../data.capnp";
using DataRef = Data.DataRef;

using Magnetics = import "../magnetics.capnp";
using MagneticField = Magnetics.MagneticField;
using Filament = Magnetics.Filament;
using BiotSavartSettings = Magnetics.BiotSavartSettings;

using Geometry = import "../geometry.capnp";
using Mesh = Geometry.Mesh;

interface CoilsDB {
	getCoil @0 (id : UInt64) -> Filament;
	getConfig @1 (id : UInt64) -> CoilsDBConfig;
}

interface ComponentsDB {
	getMesh @0 (id : UInt64) -> Mesh;
	getAssembly @1 (id : UInt64) -> (components : List(UInt64));
}

# Structure that holds instructions on how to compute the field
# for every main coil.
struct CoilFields {
	mainCoils @0 : List(DataRef(MagneticField)); # Length 7
	trimCoils @1 : List(DataRef(MagneticField)); # Length 5
	controlCoils @2 : List(DataRef(MagneticField)); # Length 10
}

# === client-side JSON interface for ComponentsDB ===

struct ComponentsDBMesh {
	surfaceMesh : group {
		nodes : group {
			x1 @0 : List(Float64);
			x2 @1 : List(Float64);
			x3 @2 : List(Float64);
		}
		polygons @3 : List(UInt32);
		numVertices @4 : List(UInt32);
	}
}

struct ComponentsDBAssembly {
	components @0 : List(UInt64);
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