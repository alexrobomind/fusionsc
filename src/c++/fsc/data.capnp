@0xa86c29cb6381b701;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

struct DataRef @0x86c1e9dee7f46dbf (T) {
	interface Getter {
		get @0 () -> (value : T);
		stream @1 (target : StreamReceiver) -> ();
	}
	interface StreamReceiver {
		receive @0 (data : Data) -> stream;
	}
	
	id     @0 : Data;
	getter @1 : Getter;
}

interface DataStore @0xc6d48902ddb7e122 {
	store @0 [T] (id : Data, data : T) -> (ref : DataRef(T));
	clone @1 [T] (source : DataRef(T)) -> (ref : DataRef(T));
	cache @2 [T] (source : DataRef(T)) -> (ref : DataRef(T));
}