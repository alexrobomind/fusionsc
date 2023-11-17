@0xa20e688ec433c35d;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Random");

struct MT19937State {
	index @0 : UInt16;
	vector @1 : List(UInt32);
}