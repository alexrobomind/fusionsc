@0x800db252e5cec051;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

import Magnetics = "magnetics.capnp";
import Geometry = "geometry.capnp";

using Data = import "data.capnp";

using DataRef = Data.DataRef;
using Float64Tensor = Data.Float64Tensor;

struct TraceRequest {
	magneticField @0 : Magnetics.MagneticField;
	startPoints @1 : Float64Tensor;
	
};