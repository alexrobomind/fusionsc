@0xc1343b1f84ad9cce;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Magnetics = import "magnetics.capnp";
using Geometry = import "geometry.capnp";
using W7X = import "devices/w7x.capnp";

struct OfflineData {
	# -- W7X ---
	
	w7xCoils @0 : List(W7XCoil);
	w7xConfigs @1 : List(W7XConfig);
	w7xComponents @2 : List(W7XComponent);
	w7xAssemblies @3 : List(W7XAssembly);
	
	struct W7XCoil {
		id @0 : UInt64;
		filament @1 : Magnetics.Filament;
	}

	struct W7XConfig {
		id @0 : UInt64;
		config @1 : W7X.CoilsDBConfig;
	}
	
	struct W7XComponent {
		id @0 : UInt64;
		component @1 : Geometry.Mesh;
	}
	
	struct W7XAssembly {
		id @0 : UInt64;
		assembly @1 : List(UInt64);
	}
}