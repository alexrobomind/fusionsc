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
	union {
		ref @1 : Data.DataRef;
		folder @2 : Folder;
	}
}

interface Folder {
	ls @0 () -> (entries : List(Text));
	getAll @1 () -> (entries : List(FolderEntry));
	getEntry @2 (name : Text) -> FolderEntry;
	putEntry @3 FolderEntry -> FolderEntry;
	
	rm @4 (name : Text) -> ();
	
	mkdir @5 (name : Text) -> (folder : Folder);
	store @6 (name : Text, ref : Data.DataRef) -> ();
}

interface Object extends(Data.DataRef, Folder) {
	enum Type { data @0; folder @1; }
	
	getInfo @0 () -> (type : Type);
}

# Internal, do not use
struct ObjectInfo {
	union {
		unresolved @0 : Void;
		exception @1 : Rpc.Exception;
		link @2 : Object;
		
		dataRef : group {
			downloadStatus : union {
				downloading @3 : Void;
				finished @4 : Void;
			}
			metadata @5 : Data.DataRefMetadata;
			capTable @6 : List(Object);
		}
		folder : group {
			entries @7 : List(FolderEntry);
		}
	}
}