@0xb4cc4351461fe90e;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::devices::w7x");

using Data = import "../data.capnp";

using Http = import "../http.capnp";
using HttpRoot = Http.HttpRoot;

using W7X = import "w7x.capnp";
using Filament = import "../magnetics.capnp".Filament;
using Mesh = import "../geometry.capnp".Mesh;

struct DBTest(T) {
	httpRoot @0 : HttpRoot;
	entries @1 : List(Entry);
	struct Entry {
		id @0 : UInt64;
		
		union {
			result @1 : T;
			notFound @2 : Void;
		}
	}
}


const cdbTest : DBTest(Filament) = (
	httpRoot = (entries = [
		(
			loc = "/coil/0/data",
			text = "{ \"polylineFilament\" : { \"vertices\" : { \"x1\" : [0.0, 3.0, 6.0], \"x2\" : [1.0, 4.0, 7.0], \"x3\" : [2.0, 5.0, 8.0] } } }"
		),
		(
			loc = "/coil/123/data",
			text = "{ \"polylineFilament\" : { \"vertices\" : { \"x1\" : [0.0, 3.0], \"x2\" : [1.0, 4.0], \"x3\" : [2.0, 5.0] } } }"
		)
	]),
	entries = [
		(
			id = 0,
			result = (
				inline = (
					shape = [3, 3],
					data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
				)
			)
		),
		(
			id = 2,
			notFound = void
		),
		(
			id = 123,
			result = (
				inline = (
					shape = [2, 3],
					data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
				)
			)
		)
	]
);

const compdbTest : DBTest(Mesh) = (
	httpRoot = (entries = [
		(
			loc = "/component/0/data",
			text = "{ \"surfaceMesh\" : { \"nodes\" : { \"x1\" : [0.0, 1.0, 1.0, 0.0], \"x2\" : [0.0, 0.0, 1.0, 1.0], \"x3\" : [0.0, 0.0, 0.0, 0.0] }, \"polygons\" : [1, 2, 3, 4], \"numVertices\" : [4] } }"
		),
		(
			loc = "/component/1/data",
			text = "{ \"surfaceMesh\" : { \"nodes\" : { \"x1\" : [0.0, 1.0, 1.0, 0.0], \"x2\" : [0.0, 0.0, 1.0, 1.0], \"x3\" : [0.0, 0.0, 0.0, 0.0] }, \"polygons\" : [1, 2, 3, 2, 3, 4], \"numVertices\" : [3, 3] } }"
		)
	]),
	entries = [
		(
			id = 0,
			result = (
				vertices = (
					shape = [4, 3],
					data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
				),
				indices = [0, 1, 2, 3],
				polyMesh = [0, 4]
			)
		),
		(
			id = 1,
			result = (
				vertices = (
					shape = [4, 3],
					data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
				),
				indices = [0, 1, 2, 1, 2, 3],
				triMesh = void
			)
		),
		(
			id = 2,
			notFound = void
		)
	]
);