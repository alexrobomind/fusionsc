@0xcc89db116b6b3313;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("cupnp::test");

struct TestStruct {
	u32 @0 : UInt32;
	d64 @3 : Float64 = 1.0;
	
	lu32 @4 : List(UInt32);
	lu32Def @5 : List(UInt32) = [4, 3, 2, 1];
	
	lb @6 : List(Bool) = [true, false, true];
	
	union1 :union {
		val1 @1 : UInt32 = 3;
		val2 @2 : UInt32 = 4;
	}
}