@0xdd4f635065dcea99;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("HTTP");

struct HttpRoot {
	struct Entry {
		loc @0 : Text;
		
		union {
			text @1 : Text;
			data @2 : Data;
		}
	}
	entries @0 : List(Entry);
}

const httpTestData : HttpRoot = (
	entries = [
		( loc = "/", text = "Hello world" ),
	]
);