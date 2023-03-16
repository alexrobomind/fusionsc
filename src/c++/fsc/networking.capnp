@0xc0fc26d592500c23;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Network");

interface NetworkInterface {
	interface Connection {
		getRemote @0 () -> (remote : Capability);
		close     @1 () -> ();
	}
	interface Listener {
		accept @0 (connection : Connection) -> (client : Capability);
	}
	
	connect    @0 (url : Text, pushInterface : Capability) -> (connection : Connection);
	listen     @1 (host : Text = "0.0.0.0", portHint : UInt16, listener : Listener) -> (assignedPort : UInt64);
	serve      @2 (host : Text = "0.0.0.0", portHint : UInt16, server : Capability) -> (assignedPort : UInt64);
	
	sshConnect @3 (host : Text, port : UInt16) -> (connection : SSHConnection);
}

interface SSHConnection extends(NetworkInterface) {
	close @0 () -> ();
	authenticatePassword @1 (user : Text, password : Text) -> ();
}