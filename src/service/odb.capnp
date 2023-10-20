@0xe3a737baddd26435;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::odb");

using Rpc = import "/capnp/rpc.capnp";

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("ODB");

using Geometry = import "geometry.capnp";
using Data = import "data.capnp";

struct FolderEntry {
	name @0 : Text;
	value @1 : Capability;
}

interface Folder {
	ls @0 (name : Text) -> (entries : List(Text));
	getAll @1 (name : Text) -> (entries : List(FolderEntry));
	getEntry @2 (name : Text) -> FolderEntry;
	putEntry @3 FolderEntry -> FolderEntry;
	
	rm @4 (name : Text) -> ();
	
	mkdir @5 (name : Text) -> (folder : Folder);
	store @6 (name : Text, ref : Data.DataRef) -> ();
}

# Internal, do not use
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
		folder : group {
			entries @8 : List(FolderEntry);
		}
	}
}