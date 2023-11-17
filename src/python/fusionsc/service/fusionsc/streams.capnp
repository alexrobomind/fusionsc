@0xf2676e4d43e04788;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Streams");

interface RemoteInputStream {
	pumpTo @0 (target : RemoteOutputStream) -> (pumpedBytes : UInt64);
	
	readAllBinary @1 () -> (data : Data);
	readAllString @2 () -> (text : Text);
}

interface RemoteOutputStream {
	write @0 (data : Data) -> stream;
	eof @1 () -> ();
	flush @2 () -> ();
}

interface RemoteIoStream extends(RemoteInputStream, RemoteOutputStream) {}