@0x91fc03b633e5bb48;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Services");

using Magnetics = import "magnetics.capnp";
using Geometry = import "geometry.capnp";
using FLT = import "flt.capnp";
using HFCam = import "hfcam.capnp";

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
	newFieldCalculator @0 (preferredDeviceType : WorkerType = gpu) -> (service : Magnetics.FieldCalculator, deviceType : WorkerType);
	newGeometryLib     @1 () -> (service    : Geometry.GeometryLib);
	newTracer          @2 (preferredDeviceType : WorkerType = gpu) -> (service    : FLT.FLT, deviceType : WorkerType);
	newHFCamProvider   @3 () -> (service : HFCam.HFCamProvider);
}