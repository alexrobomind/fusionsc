@0xe3a737baddd26435;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Rpc = import "/capnp/rpc.capnp";

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("ODB");

using Geometry = import "geometry.capnp";
using Data = import "data.capnp";

struct ODBEntry {
	union {
		unresolved @0 : Void;
		exception @1 : Rpc.Exception;
		
		resolved : union {
			notARef @2 : Void;
			downloading @3 : Void;
			downloadFailed @4 : Rpc.Exception;
			downloadSucceeded : group {
				metadata @5 : Data.DataRef(AnyPointer).Metadata;
				capTable @6 : List(Int64);
			}
		}
	}
}