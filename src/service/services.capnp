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
using Networking = import "networking.capnp";

enum ComputationDeviceType {
	cpu @0;
	gpu @1;
}

struct LocalConfig {
	preferredDeviceType @0 : ComputationDeviceType;
	enableCompute @1 : Bool = true;
	enableStorage @2 : Bool = true;
	
	jobScheduler : union {
		system @3 : Void;
		slurm @4 : Void;
	}
}

struct NodeInfo {
	deviceType @0 : ComputationDeviceType;
}

# Main service interface that can be connected to by clients.

interface RootService {
	newFieldCalculator @0 () -> (service : Magnetics.FieldCalculator);
	newGeometryLib     @1 () -> (service : Geometry.GeometryLib);
	newTracer          @2 () -> (service : FLT.FLT);
	newHFCamProvider   @3 () -> (service : HFCam.HFCamProvider);
	newKDTreeService   @4 () -> (service : Index.KDTreeService);
	newMapper          @5 () -> (service : FLT.Mapper);
	
	dataService @6 () -> (service : Data.DataService);
	
	getInfo @7 () -> NodeInfo;
}

# Default profiles
const computeNodeProfile : LocalConfig = (enableStorage = false, preferredDeviceType = gpu);
const loginNodeProfile : LocalConfig = (enableCompute = false, jobScheduler = (slurm = void));

const fscProtocolVersion : UInt64 = 1;


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

interface LocalResources extends(Networking.NetworkInterface) {
	root         @0 () -> (root : RootService);
	
	openArchive  @1 (filename : Text) -> (ref : Data.DataRef(AnyPointer));
	writeArchive @2 [T] (filename : Text, ref : Data.DataRef(T)) -> ();
	download     @3 [T] (ref : Data.DataRef(T)) -> (ref : Data.DataRef(T));
}