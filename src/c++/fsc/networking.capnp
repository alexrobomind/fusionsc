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
		unsafeCloseNow @2 () -> ();
	}
	interface Listener {
		accept @0 () -> (client : Capability);
	}
	interface OpenPort {
		getInfo @0 () -> (port : UInt64);
		
		drain @1 () -> ();
		stopListening @2 () -> ();
		
		closeAll @3 () -> ();
		unsafeCloseAllNow @4 () -> ();
	}
	
	connect    @0 (url : Text) -> (connection : Connection);
	listen     @1 (host : Text = "0.0.0.0", portHint : UInt16, listener : Listener) -> (openPort : OpenPort);
	serve      @2 (host : Text = "0.0.0.0", portHint : UInt16, server : Capability) -> (openPort : OpenPort);
	
	sshConnect @3 (host : Text, port : UInt16) -> (connection : SSHConnection);
}

interface SSHConnection extends(NetworkInterface) {
	close @0 () -> ();
	authenticatePassword @1 (user : Text, password : Text) -> ();
	authenticateKeyFile @2 (user : Text, pubKeyFile : Text, privKeyFile : Text, keyPass : Text) -> ();
}