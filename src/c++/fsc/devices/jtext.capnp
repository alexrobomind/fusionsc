@0xb7f2b56f26fc112e;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::devices::jtext");

using G = import "../geometry.capnp";

# Embed resources into the executable

const exampleGfile : Text = embed "jtext-resources/gfile.dat";
const islandCoils : List(Text) = [
	embed "jtext-resources/GP1_sco.dat",
	embed "jtext-resources/GP2_sco.dat",
	embed "jtext-resources/GP3_sco.dat",
	embed "jtext-resources/GP4_sco.dat",
	embed "jtext-resources/GP5_sco.dat",
	embed "jtext-resources/GP6_sco.dat"
];
const target : Text = embed "jtext-resources/target.dat";

const topLimiter : G.Transformed(G.Geometry) = (
	shifted = (
		shift = [0, 0, 0],
		node = ( leaf = (
			tags = [(
				name = "component", value = (text = "topLimiter" )
			)],
			wrapToroidally = (
				r = [0.925, 1.175, 1.175, 0.925, 0.925],
				z = [0.235, 0.235, 0.265, 0.265, 0.235],
				nPhi = 8,
				phiRange = (
					phiStart = ( deg = 335.5 ),
					phiEnd = ( deg = 339.5 )
				)
			)
		))
	)
);

const bottomLimiter : G.Transformed(G.Geometry) = (
	shifted = (
		shift = [0, 0, 0],
		node = ( leaf = (
			tags = [(
				name = "component", value = (text = "bottomLimiter" )
			)],
			wrapToroidally = (
				r = [0.925, 1.175, 1.175, 0.925, 0.925],
				z = [-0.235, -0.235, -0.265, -0.265, -0.235],
				nPhi = 8,
				phiRange = (
					phiStart = ( deg = 335.5 ),
					phiEnd = ( deg = 339.5 )
				)
			)
		))
	)
);

const lfsLimiter : G.Transformed(G.Geometry) = (
	shifted = (
		shift = [0, 0, 0],
		node = ( leaf = (
			tags = [(
				name = "component", value = (text = "lfsLimiter" )
			)],
			wrapToroidally = (
				r = [1.285, 1.315, 1.315, 1.285, 1.285],
				z = [-0.125, -0.125, 0.125, 0.125, -0.125],
				nPhi = 8,
				phiRange = (
					phiStart = ( deg = 335.5 ),
					phiEnd = ( deg = 339.5 )
				)
			)
		))
	)
);

const hfsLimiter : G.Geometry = (
	tags = [(
		name = "component", value = (text = "hfsLimiter" )
	)],
	wrapToroidally = (
		r = [0.7135, 0.754, 0.754, 0.7135, 0.7135],
		z = [-0.32, -0.32, 0.32, 0.32, -0.32],
		nPhi = 100
	)
);

const firstWall : G.Geometry = (
	tags = [(
		name = "component", value = (text = "firstWall" )
	)],
	wrapToroidally = (
		r = [0.7135, 1.311, 1.409,  1.409,  1.311,  0.7135, 0.7135],
		z = [0.326 , 0.326, 0.268, -0.268, -0.326, -0.326 , 0.326],
		nPhi = 100
	)
);