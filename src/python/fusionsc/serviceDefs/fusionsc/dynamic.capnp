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
		struct Field {
			name @0 : Text;
			dType @1 : DType;
			offset @2 : Int32;
		}
		
		union {
			bool @0 : Void;
			numeric : group {
				base : union {
					float @1 : Void;
					unsignedInt @2 : Void;
					signedInt @3 : Void;
					complex @4 : Void;
				}
				littleEndian @5 : Bool;
				numBytes @6 : UInt8;
			}
			special : group {
				union {
					byteArray @7 : Void;
					unicodeString @8 : Void;
				}
				littleEndian @9 : Bool;
				length @10 : UInt64;
			}
			subArray : group {
				itemType @11 : DType;
				shape @12 : List(UInt32);
			}
			struct : group {
				fields @13 : List(Field);
			}
		}
	}
	
	struct StructPointer {
		target @0 : AnyStruct;
	}
		
	struct PythonObject {
		createBy : union {
			call : group {
			    func @0 : DynamicObject;
			    args @1 : List(DynamicObject);
			}
			newobj : group {
				cls @2 : DynamicObject;
				args @3 : List(DynamicObject);
			   	kwargs @4 : List(MappingEntry);
			}
		}
		listItems @5 : List(DynamicObject);
		dictItems @6 : List(MappingEntry);
		
		state : union {
			noState @7 : Void;
			setState @8 : DynamicObject;
			withSetter : group {
				setter @9 : DynamicObject;
				value @10 : DynamicObject;
			}
		}
	}
	
	memoKey @28 : UInt64;
	
	union {
		text @0 : Text;
		
		data @1 : Data;
		bigData @2 : DataRef(Data);
		
		sequence : group {
			contents @3 : List(DynamicObject);
			kind : union {
				list @39 : Void;
				tuple @40 : Void;
				set @41 : Void;
			}
		}
		
		mapping  @4 : List(MappingEntry);
		
		ref : group {
			target @5 : DataRef;
			wrapped @6 : Bool;
		}
		
		dynamicStruct : group {
			schema @7 : AnyStruct;
			data @8 : AnyStruct;
		}
		
		dynamicCapability : group {
			schema @51 : AnyStruct;
			raw @52 : Capability;
		}
		
		uint64 @9 : UInt64;
		int64 @10 : Int64;
		
		bool @50 : Bool;
		
		double @11 : Float64;
		
		complex : group {
			real @21 : Float64;
			imag @22 : Float64;
		}
		
		dynamicEnum : group {
			schema @23 : AnyStruct;
			value  @24 : UInt16;
		}
		
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
		
		enumArray : group {
			schema @25 : AnyStruct;
			shape @26 : List(UInt64);
			data @27 : List(UInt16);
		}
		
		structArray : group {
			schema @31 : AnyStruct;
			shape @32 : List(UInt64);
			data @33 : List(StructPointer);
		}
		
		pythonBigInt @18 : Data;
		pythonPickle : union {
			data @19 : Data;
			bigData @20 : DataRef(Data);
		}
		
		# Indicates that the object is already
		# memoized under the given memo key.
		memoized @29 : Void;
		
		nested @30 : DynamicObject;
		
		# Indicates that the object is a PARENT
		# object under the given memo key
		memoizedParent @34 : Void;
		
		pythonObject @35 : PythonObject;
		
		pythonNone @36 : Void;
		
		pythonGlobal : group {
			mod @37 : Text;
			name @38 : Text;
		}
		
		unknownObject @42 : DynamicObject;
		# Object that could previously not be resolved during deserialization
		# and is being re-entered. Should be deserialized with its own memo
		# dict as the previous values are likely invalid.
		
		pythonWrapper : group {
			wrapped @43 : DynamicObject;
			
			union {
				bytes @44 : Void;
				bytearray @45 : Void;
				mutableArray @46 : Void;
				pickleBuffer @47 : Void;
			}
		}
		
		pythonNotImplemented @48 : Void;
		pythonEllipsis @49 : Void;
	}
}
