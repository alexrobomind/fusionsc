@0x91fc03b633e5bb48;

using Magnetics = import "magnetics.capnp";
using Geometry = import "geometry.capnp";

struct RootConfig {
}

enum DeviceType {
	cpu;
	gpu;
}

interface RootService {
	newFieldCalculator @0 (grid : Magnetics.ToroidalGrid, preferredDeviceType : DeviceType = gpu) -> (calculator : Magnetics.FieldCalculator, deviceType : DeviceType);
	newGeometryLib     @1 () -> (service    : Geometry.GeometryLib);
}