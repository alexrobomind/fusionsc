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
using Vmec = import "vmec.capnp";
using Matcher = import "matcher.capnp";
using Warehouse = import "warehouse.capnp";

using W7X = import "devices/w7x.capnp";

enum ComputationDeviceType {
	cpu @0;
	gpu @1;
	loop @2;
}

struct WarehouseConfig {
	name @0 : Text;
	url  @1 : Text;
	path @2 : Text = "";
}

struct LocalConfig {
	name @14 : Text = "FusionSC Server";
	
	preferredDeviceType @0 : ComputationDeviceType;
	enableCompute @1 : Bool = true;
	enableStorage @2 : Bool = true;
	
	jobScheduler : union {
		system @3 : Void;
		slurm @4 : Void;
		mpi @13 : Void;
	}
	
	# Root directory for temporary worker dirs
	jobDir @12 : Text = ".";
	
	# Configuration settings for the field line tracer
	flt @5 : FLT.FLTConfig;
	
	# Configuration for the CPU backend
	cpuBackend : group {	
		numThreads : union {
			autoDetect @6 : Void;
			fixed @7 : UInt32 = 1;
		}
	}
	
	# Configuration for the worker launcher
	workerLauncher : union {
		off @8 : Void;
		auto @9 : Void; # Only available when launching via fsc-server
		manual : group {
			coordinatorUrl @10 : Text;
		}
	}
	
	warehouses @11 : List(WarehouseConfig);
	
	loadLimit @15 : UInt32 = 100;
}

struct LoadBalancerConfig {
	struct Backend {
		url @0 : Text;
		name @1 : Text;
		persistent @2 : Bool = false;
		compressed @3 : Bool = true;
	}
	
	struct Rule {
		struct MethodSpec {
			interface @0 : UInt64;
			methods @1 : List(UInt16);
		}
		
		matches : union {
			all @0 : Void;
			only @1 : MethodSpec;
			anyOf @2 : List(MethodSpec);
			allExcept @3 : List(MethodSpec);
		}
		
		union {
			backend @4 : Backend;
			pool @5 : List(Backend);
		}
	}
	
	# backends @0 : List(Backend);
	rules @0 : List(Rule);
	
	heartbeatIntervalSeconds @1 : UInt64 = 60;
	reconnectIntervalSeconds @2 : UInt64 = 60;
}

struct NodeInfo {
	deviceType @0 : ComputationDeviceType;
	computeEnabled @1 : Bool;
	warehouses @2 : List(Text);
	name @3 : Text;
	commitHash @4 : Text;
	
	activeCalls @5 : UInt64;
	queuedCalls @6 : UInt64;
	capacity @7 : UInt64;
}

# Main service interface that can be connected to by clients.

interface RootService {
	newFieldCalculator @0 () -> (service : Magnetics.FieldCalculator);
	newGeometryLib     @1 () -> (service : Geometry.GeometryLib);
	newTracer          @2 () -> (service : FLT.FLT);
	newHFCamProvider   @3 () -> (service : HFCam.HFCamProvider);
	newKDTreeService   @4 () -> (service : Index.KDTreeService);
	newMapper          @5 () -> (service : FLT.Mapper);
	newVmecDriver      @11() -> (service : Vmec.VmecDriver);
	
	dataService @6 () -> (service : Data.DataService);
	
	matcher @8 () -> (service : Matcher.Matcher);
	
	listWarehouses @9 () -> (names : List(Text));
	getWarehouse @10 (name : Text) -> (warehouse : Warehouse.Warehouse.Folder);
	
	getInfo @7 () -> NodeInfo;
}

# Default profiles
const computeNodeProfile : LocalConfig = (enableStorage = false, preferredDeviceType = gpu);
const loginNodeProfile : LocalConfig = (enableCompute = false, jobScheduler = (slurm = void));

# Changes 1 -> 2:	Removed old field line mapping in favor of reversible mapping. This is a
#					formally incompatible change but the feature was explicitly marked as
#					unstable and experimental. The change is a pointer -> pointer change,
#					so the protocol remains fully compatible in all other aspects.
const fscProtocolVersion : UInt64 = 2;


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
	
	configureRoot @4 LocalConfig -> ();
	
	w7xProvider @5 () -> (service : W7X.Provider);
	
	openWarehouse @6 (url : Text, networkInterface : Networking.NetworkInterface) -> (object : Warehouse.Warehouse.GenericObject, storedObject : Warehouse.Warehouse.StoredObject);
}
