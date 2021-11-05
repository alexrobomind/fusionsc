@0xa86c29cb6381b701;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

interface DataRef (T) {	
	metadata @0 () -> (
		id : Data,
		typeId : UInt64,
		capTableSize : UInt64
	);
	
	rawBytes @1 () -> (data : Data);
	capTable @2 () -> (table : List(Capability));
}

interface DataStore @0xc6d48902ddb7e122 {
	store @0 [T] (id : Data, data : T) -> (ref : DataRef(T));
	clone @1 [T] (source : DataRef(T)) -> (ref : DataRef(T));
	cache @2 [T] (source : DataRef(T)) -> (ref : DataRef(T));
}