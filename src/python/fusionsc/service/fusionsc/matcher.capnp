@0xd9594a6fca7b40b2;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Matcher");

interface Matcher {
	get @0 () -> (token : Data, cap : Capability);
	put @1 (token : Data, cap : Capability) -> ();
}