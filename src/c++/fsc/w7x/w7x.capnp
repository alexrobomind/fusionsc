@0xc4b38b12cd771bac;

interface CoilsDB {
	getCoil @0 (id : UInt64) -> (filament : DataRef<Filament>);
}

struct UnresolvedCoilPack {
	width @0 : Float64;
	coils @1 : W7XCoilSet;
}

struct ResolvedCoilPack {
	mainCoils : List(DataRef(MagneticField));
	trimCoils : List(DataRef(MagneticField));
	controlCoils : List(DataRef(MagneticField));
}