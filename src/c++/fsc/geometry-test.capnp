@0xdcc55edb4a648dcb;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Geo = import "geometry.capnp";

const testCube : Geo.Mesh = (
	vertices = (
		shape = [8, 3],
		data = [
			-1, -1, -1, # 0
			 1, -1, -1, # 1
			 1,  1, -1, # 2
			-1,  1, -1, # 3
			-1, -1,  1, # 4
			 1, -1,  1, # 5
			 1,  1,  1, # 6
			-1,  1,  1  # 7
		]
	),
	
	indices = [
		# xy surfaces
		0, 1, 2, 3,
		4, 5, 6, 7,
		
		# xz surfaces
		0, 4, 5, 1,
		2, 6, 7, 3,
		
		# yz surfaces
		0, 3, 7, 4,
		1, 5, 6, 2
	],
	
	polyMesh = [0, 4, 8, 12, 16, 20, 24]
);