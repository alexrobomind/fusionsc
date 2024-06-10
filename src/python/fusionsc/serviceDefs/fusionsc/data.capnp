@0xa86c29cb6381b701;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Rpc = import "/capnp/rpc.capnp";

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Data");

struct DataRefMetadata {
	id @0 : Data;
	capTableSize @1 : UInt64;
	dataSize @2 : UInt64;
	dataHash @3 : Data;
	
	format : union {
		unknown @4 : Void;
		schema @5 : AnyPointer; # In reality, this is capnp/schema.Type, but we don't want to include that file here
		raw @6 : Void;
	}
}

interface DataRef (T) {
	interface Receiver {
		begin @0 (numBytes : UInt64) -> ();
		receive @1 (data : Data) -> stream;
		done @2 () -> ();
	}
	
	metaAndCapTable @0 () -> (metadata : DataRefMetadata, table : List(Capability));
	rawBytes @1 (start : UInt64, end : UInt64) -> (data : Data);
	transmit @2 (start : UInt64, end : UInt64, receiver : Receiver);
}

#//! [DataService]
interface DataService @0xc6d48902ddb7e122 {
	# Upload a message to the remote data service and have it publish it
	store @0 [T] (id : Data, data : T, schema : AnyPointer) -> (ref : DataRef(T));
	
	# Have the remote data service download a DataRef and re-publish it
	clone @1 [T] (source : DataRef(T)) -> (ref : DataRef(T));
	
	# Have the remote service inspect the linked tree of refs and their content tables and
	# compute a hash based on the received information.
	hash @2 [T] (source : DataRef(T)) -> (hash : Data);
	
	# Have the remote data service download a complete copy
	cloneAllIntoMemory @3 [T] (source : DataRef(T)) -> (ref : DataRef(T));
}
#//! [DataService]

# Archive types
#struct Archive {
#	struct CapabilityInfo {
#		dataRefInfo : union {
#			noDataRef @0 : Void;
#			refID @1 : Data;
#		}
#	}
#	struct Entry {
#		id @0 : Data;
#		data @1 : List(Data);
#		capabilities @2 : List(CapabilityInfo);
#		typeId @3 : UInt64;
#		dataHash @4 : Data;
#	}
#	root @0 : Entry;
#	extra @1 : List(Entry);
#}

# Support data types

# BEGIN [tensors]

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

# END [tensors]