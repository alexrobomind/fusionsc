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
	
	rawBytes @1 (start : UInt64, end : UInt64) -> (data : Data);
	capTable @2 () -> (table : List(Capability));
}

interface DataService @0xc6d48902ddb7e122 {
	store @0 [T] (id : Data, data : T) -> (ref : DataRef(T));
	clone @1 [T] (source : DataRef(T)) -> (ref : DataRef(T));
	cache @2 [T] (source : DataRef(T)) -> (ref : DataRef(T));
}

# Archive types
struct Archive {
	struct CapabilityInfo {
		dataRefInfo : union {
			noDataRef @0 : Void;
			refID @1 : Data;
		}
	}
	struct Entry {
		id @0 : Data;
		data @1 : List(Data);
		capabilities @2 : List(CapabilityInfo);
		typeId @3 : UInt64;
	}
	root @0 : Entry;
	extra @1 : List(Entry);
}

# Support data types

struct Float64Tensor {
	shape @0 : List(UInt64);
	data  @1 : List(Float64);
}

struct Float32Tensor {
	shape @0 : List(UInt64);
	data  @1 : List(Float32);
}

struct Int64Tensor {
	shape @0 : List(UInt64);
	data  @1 : List(Int64);
}

struct UInt64Tensor {
	shape @0 : List(UInt64);
	data  @1 : List(UInt64);
}

struct Int32Tensor {
	shape @0 : List(UInt64);
	data  @1 : List(Int32);
}

struct UInt32Tensor {
	shape @0 : List(UInt64);
	data  @1 : List(UInt32);
}

struct ShapedList(ListType) {
	shape @0 : List(UInt64);
	data  @1 : ListType;
}