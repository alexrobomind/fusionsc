#pragma once

#include "common.h"
#include "eigen.h"
#include "data.h"
#include "kernels/device.h"

#include <fsc/geometry.capnp.h>

/**
 * \defgroup Geometry
 * @{
 *
 * Several subfunctions of FusionSC require the specification of machine geometries. To support rich geometry
 * representations, FusionSC offers a high-level geometry representation based on both meshes, declarative
 * operations (2D -> 3D extrusions, geometric operations on other geometries), and labeling support.
 *
 * ### Geometry processing
 *
 * Geometry processing is primarily handled through the fsc::GeometryLib service interface. An instance of this
 * interface can be obtained through fsc::newGeometryLib.
 *
 * \snippet geometry.capnp GeoLib
 *
 * ### Geometry representation
 *
 * Geometries are fully represented using Cap'n'proto types. 
 *
 * \snippet geometry.capnp Geometry
*/

namespace fsc {

//! Checks whether a geometry node is a builtin node that can be directly processed by the geometry resolver.
bool isBuiltin(Geometry::Reader);
	
struct GeometryResolverBase : public GeometryResolver::Server {
	Promise<void> resolveGeometry(ResolveGeometryContext context) override;
	
	virtual Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context);
	        Promise<void> processTransform(Transformed<Geometry>::Reader input, Transformed<Geometry>::Builder output, ResolveGeometryContext context);
};

/**
 * Creates C++ interface to geometry library.
 */
Own<GeometryLib::Server> newGeometryLib(Own<DeviceBase>);

/**
 * @}
*/

Vec3u locationInGrid(Vec3d point, CartesianGrid::Reader reader);

inline EIGEN_DEVICE_FUNC Vec3u locationInGrid(Vec3d point, Vec3d min, Vec3d max, Vec3u size) {
	auto fraction = (point - min).array() / (max - min).array();
	auto perCell = fraction * size.array().cast<double>();
	Vec3i result = perCell.cast<int>();
	
	for(size_t i = 0; i < 3; ++i) {
		if(result[i] < 0)
			result[i] = 0;
		
		if(result[i] >= size[i])
			result[i] = size[i] - 1;
	}
	
	return result.cast<unsigned int>();
}

Temporary<Geometry> readPly(kj::StringPtr filename);

void writePly(MergedGeometry::Reader, kj::StringPtr filename, bool binary);

void importRaw(kj::ArrayPtr<std::array<const double, 3>> vertices, kj::ArrayPtr<kj::Array<const size_t>> faces, MergedGeometry::Builder out);
kj::Tuple<kj::Array<std::array<double, 3>>, kj::Array<kj::Array<size_t>>> exportRaw(MergedGeometry::Reader, bool triangulate);

Mat4d rotationAxisAngle(Vec3d center, Vec3d axis, double angle);
double angle(Angle::Reader);

}
