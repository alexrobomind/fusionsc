@0xa86c29cb6381b701;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

interface DataRef (T) {
	struct Metadata {
		id @0 : Data;
		typeId @1 : UInt64;
		capTableSize @2 : UInt64;
		dataSize @3 : UInt64;
	}
	
	metadata @0 () -> (
		metadata : Metadata
	);
	
	rawBytes @1 () -> (data : Data);
	capTable @2 () -> (table : List(Capability));
}

interface DataService @0xc6d48902ddb7e122 {
	store @0 [T] (id : Data, data : T) -> (ref : DataRef(T));
	clone @1 [T] (source : DataRef(T)) -> (ref : DataRef(T));
	cache @2 [T] (source : DataRef(T)) -> (ref : DataRef(T));
}