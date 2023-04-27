@0xc2937719234fa6c2;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc");

using Java = import "java.capnp";
$Java.package("org.fsc");
$Java.outerClassname("HFCam");

using Geometry = import "geometry.capnp";
using Data = import "data.capnp";

struct HFCamProjection {
	width  @0 : UInt32;
	height @1 : UInt32;
	
	transform @2 : List(Float64); # 4x4 column-major transform matrix into homogenous coordinates
	clipPlanes @3 : List(List(Float64)); # [a, b, c, d]: Clip volumes of the form ax + by + cz + d > 0
}

struct HFCamData {
	projection @0 : HFCamProjection;
	
	determinantBuffer @1 : Data.Float64Tensor;
	depthBuffer @2 : Data.Float64Tensor;
}

interface HFCamProvider {
	makeCamera @0 (projection: HFCamProjection, geometry : Geometry.Geometry, edgeTolerance : Float64 = 0.5, depthTolerance : Float64) -> HFCamData;
	makeToroidalProjection @1 (
		w: UInt32,
		h : UInt32,
		phi : Float64,
		rTarget : Float64, zTarget: Float64,
		inclination: Float64, horizontalInclination: Float64, distance: Float64,
		viewportHeight: Float64, fieldOfView: Float64
	) -> HFCamProjection;
}