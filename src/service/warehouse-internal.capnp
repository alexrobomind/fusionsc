@0xd800185fcb6d4c6d;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::internal");

using Rpc = import "/capnp/rpc.capnp";

using Data = import "data.capnp";

# This class is used for internal storage inside the warehouse. Do not use.
struct ObjectInfo {	
	union {
		unresolved @0 : Void;
		nullValue @1 : Void;
		exception @2 : Rpc.Exception;
		link @3 : Capability;
		
		dataRef : group {
			downloadStatus : union {
				downloading @4 : Void;
				finished @5 : Void;
			}
			metadata @6 : Data.DataRefMetadata;
			capTable @7 : List(Capability);
		}
		folder @8 : Void; # Folder links are implemented in the database for indexing
		file @9 : Capability;
	}
}