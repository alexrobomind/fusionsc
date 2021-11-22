@0xdd4f635065dcea99;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

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