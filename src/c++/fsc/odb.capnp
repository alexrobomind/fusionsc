@0xe3a737baddd26435;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::odb");

using Rpc = import "/capnp/rpc.capnp";

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("ODB");

using Geometry = import "geometry.capnp";
using Data = import "data.capnp";

FolderEntry {
	name @0 : Text;
	union {
		ref @1 : DataRef<AnyPointer>();
		folder @2 : Folder;
	}
};

struct FolderData {
	struct Entry {
		name @0 : Text;
		ref @1 : Object(AnyPointer);
	}
	
	entries @0 : List(Entry);
}

struct ObjectInfo {
	union {
		unresolved @0 : Void;
		exception @1 : Rpc.Exception;
		dataRef : union {
			downloadStatus : union {
				downloading @2 : Void;
				finished @3 : Void;
			}
			metadata @4 : Data.DataRef(AnyPointer).Metadata;
			capTable @5 : List(ODBObject(AnyPointer));
		}
		folder @6 : FolderData;
	}
}

interface Folder {
	ls @0 () -> (entries : List(Text));
	getAll @1 () -> (entries : List(FolderEntry));
	getEntry @2 (name : Text) -> FolderEntry;
	putEntry @3 FolderEntry -> ();
	
	mkdir @4 (name : Text) -> (folder : Folder);
	store @5 (name : Text, ref : DataRef<AnyPointer>) -> ();
}

interface Object extends(Data.DataRef(AnyPointer), Folder) {
	enum Type { data @0; folder @1; }
	
	getInfo @0 : () -> (type : Type);
}