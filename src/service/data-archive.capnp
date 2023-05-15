@0xe504565ca30d34fe;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("DataArchive");

using Rpc = import "/capnp/rpc.capnp";
using Data = import "data.capnp";


struct ArchiveInfo {
	struct RefInfo {
		union {
			null @0 : Void;
			exception @1 : Rpc.Exception;
			object @2 : UInt64;
		}
	}
	struct ObjectInfo {
		metadata @0 : Data.DataRefMetadata;
		dataId @1 : UInt64;
		refs @2 : List(RefInfo);
	}
	struct DataInfo {
		offsetWords @0 : UInt64;
		sizeBytes @1 : UInt64;
	}
	
	objects @0 : List(ObjectInfo);
	data @1 : List(DataInfo);
	root @2 : UInt64;
}