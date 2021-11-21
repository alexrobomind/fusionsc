@0xc4b38b12cd771bac;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::devices::w7x");

interface CoilsDB {
	getCoil @0 (id : UInt64) -> (filament : DataRef<Filament>);
}

# Structure that holds instructions on how to compute the field
# for every main coil.
struct CoilFields {
	mainCoils : List(DataRef(MagneticField));
	trimCoils : List(DataRef(MagneticField));
	controlCoils : List(DataRef(MagneticField));
}