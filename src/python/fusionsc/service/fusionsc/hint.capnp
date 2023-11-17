@0xcdf88e793a1272ca;

using D = import "data.capnp";
using M = import "magnetics.capnp";

struct HintEquilibrium {
	grid @0 : M.ToroidalGrid;
	
	# Data shape is [phi, z, r, 3], C (row-major) ordering with last index stride 1
	# When interpreting column-major, data shape is [3, r, z, phi]
	# The field components are ordered Phi, Z, R along the smallest-stride (stride 1) axis
	field @1 : D.DataRef(D.Float64Tensor);
	
	# Data shape is [phi, z, r]
	pressure @2 : D.DataRef(D.Float64Tensor);
	
	# Data shape is [phi, z, r, 3]
	# The field components are ordered Phi, Z, R along the smallest-stride (stride 1) axis
	velocity @3 : D.DataRef(D.Float64Tensor);
}