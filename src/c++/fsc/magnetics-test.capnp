@0xc3eb8f57e6b4ef3c;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using MagneticField = import "magnetics.capnp".MagneticField;
using ToroidalGrid = import "magnetics.capnp".ToroidalGrid;

const testGrid : ToroidalGrid = (
	rMin = 0.2,
	rMax = 1.0,
	zMin = -1,
	zMax = 1,
	nSym = 1,
	nR = 4,
	nZ = 10,
	nPhi = 16
);

const wireField : MagneticField = (
	filamentField = (
		current = 1.0,
		biotSavartSettings = (
			stepSize = 0.01
		),
		windingNo = 1,
		filament = (
			inline = (
				shape = [2, 3],
				data = [
					0, 0, -1,
					0, 0, 1
				]
			)
		)
	)
);