@0x91fc03b633e5bb48;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("Services");

using Magnetics = import "magnetics.capnp";
using Geometry = import "geometry.capnp";
using FLT = import "flt.capnp";
using HFCam = import "hfcam.capnp";
using Index = import "index.capnp";
using Data = import "data.capnp";

struct RootConfig {
}

enum WorkerType {
	cpu @0;
	gpu @1;
}

interface RootService {
	newFieldCalculator @0 (preferredDeviceType : WorkerType = gpu) -> (service : Magnetics.FieldCalculator, deviceType : WorkerType);
	newGeometryLib     @1 () -> (service    : Geometry.GeometryLib);
	newTracer          @2 (preferredDeviceType : WorkerType = gpu) -> (service    : FLT.FLT, deviceType : WorkerType);
	newHFCamProvider   @3 () -> (service : HFCam.HFCamProvider);
	newKDTreeService   @4 () -> (service : Index.KDTreeService);
	newMapper          @5 () -> (service : FLT.Mapper);
	
	dataService @6 () -> (service : Data.DataService);
}

interface NetworkInterface {
	connect    @0 (url : Text) -> (client : Capability);
	sshConnect @1 (host : Text, port : UInt16) -> (connection : SSHConnection);
}

interface SSHConnection extends(NetworkInterface) {
	close @0 () -> ();
	authenticatePassword @1 (user : Text, password : Text) -> (success : Bool);
}

# Extended local interface to provide access to the local file system and network connections
#
# This interface is intended to support interactive applications where the event loop in the main
# thread might be blocked for an extended amount of time. This interface allows another thread to
# proxy capabilities based on network connections & the file system to allow concurrent access
# to these resources, which is important if the local node acts as a hub connecting multiple
# different resources or exposes data from its local filesystem that might be needed with
# a delay.
# This interface is exposed through the LocalWorker class, that maintains the facilities to
# connect to it from multiple threads.
#
# !!! NEVER EXPOSE THIS INTERFACE EXTERNALLY !!!

interface LocalResources extends(NetworkInterface) {
	root         @0 () -> (root : RootService);
	
	openArchive  @1 (filename : Text) -> (ref : Data.DataRef);
	writeArchive @2 (filename : Text, ref : Data.DataRef) -> ();
	download     @3 (ref : Data.DataRef) -> (ref : Data.DataRef);
}