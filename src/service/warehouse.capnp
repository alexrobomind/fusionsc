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
		asFolder @0 : Folder;
		asRef    @1 : Data.DataRef;
		asFile   @2 : File;
		
		union {
			unresolved @3 : Void;
			nullValue  @4 : Void;
			exception  @5 : Rpc.Exception;
			folder     @6 : Void;
			file       @7 : Void;
			dead       @8 : Void;
			
			dataRef  : group {
				downloadStatus : union {
					downloading @9 : Void;
					finished @10 : Void;
				}
			}
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
	}
	
	interface File(StaticType) {
		set @0 (ref : Data.DataRef(StaticType)) -> ();
		get @1 () -> (ref : Data.DataRef(StaticType));
		
		setAny @2 (value : Capability) -> ();
		getAny @3 () -> StoredObject;
	}
	
	getRoot @0 () -> (root : Folder);
}