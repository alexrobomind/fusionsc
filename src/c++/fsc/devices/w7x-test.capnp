@0xb4cc4351461fe90e;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::devices::w7x");

using Data = import "../data.capnp";

using Http = import "../http.capnp";
using HttpRoot = Http.HttpRoot;

using W7X = import "w7x.capnp";
using Filament = import "../magnetics.capnp".Filament;

struct CoilsDBTest {
	httpRoot @0 : HttpRoot;
	entries @1 : List(Entry);
	struct Entry {
		id @0 : UInt64;
		
		union {
			result @1 : Filament;
			notFound @2 : Void;
		}
	}
}


const cdbTest : CoilsDBTest = (
	httpRoot = (entries = [
		(
			loc = "/data/0",
			text = "{ \"polylineFilament\" : { \"x1\" : [0.0, 3.0, 6.0], \"x2\" : [1.0, 4.0, 7.0], \"x3\" : [2.0, 5.0, 8.0] } }"
		),
		(
			loc = "/data/123",
			text = "{ \"polylineFilament\" : { \"x1\" : [0.0, 3.0], \"x2\" : [1.0, 4.0], \"x3\" : [2.0, 5.0] } }"
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