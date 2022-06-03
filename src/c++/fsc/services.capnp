@0x91fc03b633e5bb48;

using Magnetics = import "magnetics.capnp";

struct RootConfig {
}

interface RootService {
	newFieldCalculator @0 () -> (calculator : Magnetics.FieldCalculator);
}