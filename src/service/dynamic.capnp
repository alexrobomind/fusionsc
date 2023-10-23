@0x9336b515b7d9a171;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Dynamic");

using DataRef = import "data.capnp".DataRef;

struct DynamicObject {
	struct MappingEntry {
		key   @0 : DynamicObject;
		value @1 : DynamicObject;
	}
	
	struct DType {
		union {
			bool @0 : Void;
			numeric : group {
				base : union {
					float @1 : Void;
					unsignedInt @2 : Void;
					signedInt @3 : Void;
				}
				littleEndian @4 : Bool;
				numBytes @5 : UInt8;
			}
		}
	}
	
	union {
		text @0 : Text;
		data @1 : Data;
		bigData @2 : DataRef(Data);
		
		sequence @3 : List(DynamicObject);
		mapping  @4 : List(MappingEntry);
		
		ref : group {
			target @5 : DataRef(AnyPointer);
			wrapped @6 : Bool;
		}
		
		dynamicStruct : group {
			schema @7 : AnyStruct;
			data @8 : AnyStruct;
		}
		
		uint64 @9 : UInt64;
		int64 @10 : Int64;
		
		double @11 : Float64;
		
		array : group {
			dType @12 : DType;
			shape @13 : List(UInt64);
			
			union {
				data @14 : Data;
				bigData @15 : DataRef(Data);
			}
		}
		
		dynamicObjectArray : group {
			shape @16 : List(UInt64);
			data @17 : List(DynamicObject);
		}
		
		pythonBigInt @18 : Data;
		pythonPickle : union {
			data @19 : Data;
			bigData @20 : DataRef(Data);
		}
	}
}