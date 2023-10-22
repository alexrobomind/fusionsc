@0x9336b515b7d9a171;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Dynamic");

using DataRef = import "data.capnp".DataRef;

struct DynamicObject {
	struct MappingEntry {
		key : DynamicObject;
		value : DynamicObject;
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
		bigData @15 : DataRef(Data);
		
		sequence @2 : List(DynamicObject);
		mapping  @3 : List(MappingEntry);
		
		ref : group {
			target @4 : DataRef(AnyPointer);
			wrapped @16 : Bool;
		}
		
		dynamicStruct : group {
			schema @5 : AnyStruct;
			data @6 : AnyStruct;
		}
		
		uint64 @4 : UInt64;
		int64 @5 : Int64;
		
		double @6 : Double;
		
		array : group {
			dType @7 : DType;
			shape @8 : List(UInt64);
			
			union {
				data @9 : Data;
				bigData @17 : DataRef(Data);
			}
		}
		
		dynamicObjectArray : group {
			shape @11 : List(UInt64);
			data @12 : List(DynamicObject);
		}
		
		pythonBigInt @13 : Data;
		pythonPickle : union {
			data @14 : Data;
			bigData @18 : DataRef(Data);
		}
	}
};