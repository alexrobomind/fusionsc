struct HFCamProjection {
	width  @0 : Float64;
	height @1 : Float64;
	
	transform @2 : List(Float64); # 4x4 column-major transform matrix into homogenous coordinates
	clipPlanes @3 : List(List(Float64)); # [a, b, c, d]: Clip volumes of the form ax + by + cz + d > 0
}