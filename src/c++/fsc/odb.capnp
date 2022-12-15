@0xe3a737baddd26435;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::odb");

using Rpc = import "/capnp/rpc.capnp";

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("ODB");

using Geometry = import "geometry.capnp";
using Data = import "data.capnp";

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

struct ObjectEntry {
	struct RefInfo {
		union {
			null @0 : Void;
			error @1 : Rpc.Exception;
			link @2 : Int64;
		}
	}
}
	info @0 : ObjectInfo;
	refs @1 : List(RefInfo);
}

interface Folder {
	ls @0 () -> (entries : List(Text));
	getAll @1 () -> (entries : List(FolderData.Entry));
	getObject @2 (name : Text) -> (object : Object(AnyPointer));
	setObject @3 (name : Text, object : AnyPointer) -> ();
}

interface Object extends(Data.DataRef(AnyPointer), Folder {
	enum Type { data @0; folder @1; }
	
	getInfo @0 : () -> (type : Type);
}