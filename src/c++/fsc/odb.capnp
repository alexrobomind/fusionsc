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
	
	mkdir @4 (name : Text) -> (folder : Folder);
	store @5 (name : Text, ref : Data.DataRef) -> ();
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
		dataRef : union {
			downloadStatus : union {
				downloading @2 : Void;
				finished @3 : Void;
			}
			metadata @4 : Data.DataRef.Metadata;
			capTable @5 : List(Object);
		}
		folder : group {
			entries @6 : List(FolderEntry);
		}
	}
}