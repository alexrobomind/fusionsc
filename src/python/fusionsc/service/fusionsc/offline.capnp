@0xc1343b1f84ad9cce;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Offline");

using M = import "magnetics.capnp";
using G = import "geometry.capnp";

struct OfflineData {
	struct Mapping(T) {
		key @0 : T;
		val @1 : T;
	}
	
	coils @0 : List(Mapping(M.Filament));
	fields @1 : List(Mapping(M.MagneticField));
	geometries @2 : List(Mapping(G.Geometry));
}