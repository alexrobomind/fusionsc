@0xad5af37264f2cf6c;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Jobs");

using Streams = import "streams.capnp";

interface Job {
	enum State {
		pending @0;
		running @1;
		failed @2;
		completed @3;
	}
	
	struct AttachResponse {
		stdin @0 : Streams.RemoteOutputStream;
		stdout @1 : Streams.RemoteInputStream;
		stderr @2 : Streams.RemoteInputStream;
	}
	
	getState @0 () -> (state : State);
	cancel @1 () -> ();
	detach @2 () -> ();
	
	whenRunning @3 () -> ();
	whenCompleted @4 () -> ();
	
	attach @5 () -> AttachResponse;
	eval @6 (stdIn : Data) -> (stdOut : Text, stdErr : Text);
}