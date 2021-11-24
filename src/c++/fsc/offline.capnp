@0xc1343b1f84ad9cce;

using Magnetics = import "magnetics.capnp";
using W7X = import "devices/w7x.capnp";

struct OfflineData {
	# -- W7X ---
	w7xCoils @0 : List(W7XCoil);
	w7xConfigs @1 : List(W7XConfig);
	
	struct W7XCoil {
		id @0 : UInt64;
		filament @1 : Magnetics.Filament;
	}

	struct W7XConfig {
		id @0 : UInt64;
		config @1 : W7X.CoilsDBConfig;
	}
}