@0xe3a737baddd26435;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Rpc = import "/capnp/rpc.capnp";

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Warehouse");

using Data = import "data.capnp";

interface Warehouse {	
	struct StoredObject {
		
		asGeneric @0 : GenericObject;
		
		union {
			unresolved @1 : Void;
			nullValue  @2 : Void;
			exception  @3 : Rpc.Exception;
			folder     @4 : Folder;
			file       @5 : File;
			dead       @6 : Void;
			
			dataRef  : group {
				downloadStatus : union {
					downloading @7 : Void;
					finished @8 : Void;
				}
				asRef @9 : Data.DataRef(AnyPointer);
			}
		}
	}
	
	struct FrozenFolder {
		struct Entry {
			name @0 : Text;
			value @1 : FrozenEntry;
		}
		
		entries @0 : List(Entry);
	}
	
	struct FrozenEntry {
		union {
			unavailable @0 : Void;
			folder @1 : FrozenFolder;
			file : union {
				nullValue @2 : Void;
				value @3 : FrozenEntry;
			}
			dataRef @4 : Data.DataRef;
		}
	}
	
	interface Folder {
		struct Entry {
			name @0 : Text;
			value @1 : StoredObject;
		}
		
		ls @0 (path : Text) -> (entries : List(Text));
		getAll @1 (path : Text) -> (entries : List(Entry));
		get @2 (path : Text) -> StoredObject;
		put @3 (path : Text, value : Capability) -> StoredObject;
		
		rm @4 (path : Text) -> ();
		
		mkdir @5 (path : Text) -> (folder : Folder);
		
		createFile @6 (path : Text) -> (file : File);
		
		freeze @7 (path : Text) -> (ref : Data.DataRef(FrozenFolder));
	}
	
	interface File(StaticType) {
		set @0 (ref : Data.DataRef(StaticType)) -> ();
		get @1 () -> (ref : Data.DataRef(StaticType));
		
		setAny @2 (value : Capability) -> ();
		getAny @3 () -> StoredObject;
	}
	
	interface GenericObject extends (Data.DataRef(AnyPointer), Folder, File) {
	}
	
	getRoot @0 (name : Text = "root") -> (root : Folder);
}