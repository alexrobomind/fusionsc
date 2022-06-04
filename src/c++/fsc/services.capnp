@0x91fc03b633e5bb48;

using Magnetics = import "magnetics.capnp";
using Geometry = import "geometry.capnp";

struct RootConfig {
}

interface RootService {
	newFieldCalculator @0 () -> (calculator : Magnetics.FieldCalculator);
	newGeometryLib     @1 () -> (service    : Geometry.GeometryLib);
}