#pragma once

#include "common.h"
#include "eigen.h"
#include "data.h"

#include <fsc/geometry.capnp.h>

namespace fsc {

bool isBuiltin(Geometry::Reader);
	
struct GeometryResolverBase : public GeometryResolver::Server {
	Promise<void> resolveGeometry(ResolveGeometryContext context) override;
	
	virtual Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context);
	        Promise<void> processTransform(Transformed<Geometry>::Reader input, Transformed<Geometry>::Builder output, ResolveGeometryContext context);
};

/**
 * Creates C++ interface to geometry library.
 */
GeometryLib::Client newGeometryLib();

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

Promise<void> writePly(Geometry::Reader, kj::StringPtr filename, bool binary = true);
void writePly(MergedGeometry::Reader, kj::StringPtr filename, bool binary);

void importRaw(kj::ArrayPtr<std::array<const double, 3>> vertices, kj::ArrayPtr<kj::Array<const size_t>> faces, MergedGeometry::Builder out);
kj::Tuple<kj::Array<std::array<double, 3>>, kj::Array<kj::Array<size_t>>> exportRaw(MergedGeometry::Reader, bool triangulate);

}