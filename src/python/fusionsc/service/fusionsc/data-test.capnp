@0xfb6666e6e3d75673;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::test");

# This file contains additional data structures to help with data testing

using import "data.capnp".DataRef;

struct DataHolder {
	data @0 : Data;
}

struct DataRefHolder(T) {
	ref @0 : DataRef(T);
}

interface A {}
interface B extends(A) {}

struct TestStruct {
	void @0 : Void;
	bool @1 : Bool;
	
	ints : union {
		int8 @2 : Int8;
		int16 @3 : Int16;
		int32 @4 : Int32;
		int64 @5 : Int64;
	}
	
	uints : union {
		uint8 @6 : UInt8;
		uint16 @7 : UInt16;
		uint32 @8 : UInt32;
		uint64 @9 : UInt64;
	}
	
	pointers : group {
		data @10 : Data;
		text @11 : Text;
	}
}

const test0 : TestStruct = ();
const test1 : TestStruct = ( bool = true );

const test2 : TestStruct = ( ints = ( int8 = 1 ) );
const test3 : TestStruct = ( ints = ( int16 = 1 ) );
const test4 : TestStruct = ( ints = ( int32 = 1 ) );
const test5 : TestStruct = ( ints = ( int64 = 1 ) );

const test6 : TestStruct = ( uints = ( uint8 = 1 ) );
const test7 : TestStruct = ( uints = ( uint16 = 1 ) );
const test8 : TestStruct = ( uints = ( uint32 = 1 ) );
const test9 : TestStruct = ( uints = ( uint64 = 1 ) );

const test10 : TestStruct = ( pointers = (data = 0x"9f") );
const test11 : TestStruct = ( pointers = (text = "Hi") );