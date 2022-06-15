@0x91fc03b633e5bb48;

using Magnetics = import "magnetics.capnp";
using Geometry = import "geometry.capnp";
using FLT = import "flt.capnp";

struct RootConfig {
}

enum WorkerType {
	cpu @0;
	gpu @1;
}

interface ResolverChain extends(Magnetics.FieldResolver, Geometry.GeometryResolver) {
	register @0 (resolver : Capability) -> (registration : Capability);
}

interface RootService {
	newFieldCalculator @0 (grid : Magnetics.ToroidalGrid, preferredDeviceType : WorkerType = gpu) -> (calculator : Magnetics.FieldCalculator, deviceType : WorkerType);
	newGeometryLib     @1 () -> (service    : Geometry.GeometryLib);
	newTracer          @2 () -> (service    : FLT.FLT);
}