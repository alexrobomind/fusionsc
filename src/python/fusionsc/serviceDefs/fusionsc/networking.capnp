@0xc0fc26d592500c23;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Network");

interface SimpleHttpServer {
	struct Request {
		method @0 : Text;
		url @1 : Text;
	}

	struct Response {
		status @0 : UInt16;
		body @1 : Text;
		statusText @2 : Text;
	}
	
	serve @0 Request -> Response;
}

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
	
	connect    @0 (url : Text, allowCompression : Bool = true) -> (connection : Connection);
	listen     @1 (host : Text = "0.0.0.0", portHint : UInt16, listener : Listener, fallback : SimpleHttpServer) -> (openPort : OpenPort);
	serve      @2 (host : Text = "0.0.0.0", portHint : UInt16, server : Capability, fallback : SimpleHttpServer) -> (openPort : OpenPort);
	
	sshConnect @3 (host : Text, port : UInt16) -> (connection : SSHConnection);
}

interface SSHConnection extends(NetworkInterface) {
	close @0 () -> ();
	authenticatePassword @1 (user : Text, password : Text) -> ();
	authenticateKeyFile @2 (user : Text, pubKeyFile : Text, privKeyFile : Text, keyPass : Text) -> ();
	
	# This function is not yet available due to a bug in the libssh2 library preventing encrypted keys
	# in in-memory PEM data.
	# See https://github.com/libssh2/libssh2/issues/1047
	# authenticateKeyData @3 (user : Text, pubKey : Text, privKey : Text, keyPass : Text) -> ();
}
