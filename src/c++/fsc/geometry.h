#pragma once

#include "local.h"
#include "data.h"
#include "tensor.h"

#include <cupnp/cupnp.h>

#include <fsc/geometry.capnp.h>
#include <fsc/geometry.capnp.cu.h>

namespace fsc {
	
struct GeometryResolverBase : public GeometryResolver::Server {
	LibraryThread lt;
	GeometryResolverBase(LibraryThread& lt);
	
	virtual Promise<void> resolve(ResolveContext context) override;
	
	virtual Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveContext context);
	        Promise<void> processTransform(Transformed<Geometry>::Reader input, Transformed<Geometry>::Builder output, ResolveContext context);
};

struct GeometryLibImpl : public GeometryLib::Server {
	LibraryThread lt;
	GeometryLibImpl(LibraryThread& lt);
	
	Promise<void> merge(MergeContext context) override;
	Promise<void> index(IndexContext context) override;
	
private:
	struct GeometryAccumulator {
		kj::Vector<Temporary<MergedGeometry::Entry>> entries;
		
		inline void finish(MergedGeometry::Builder output) {
			auto outEntries = output.initEntries(entries.size());
			for(size_t i = 0; i < entries.size(); ++i) {
				outEntries.setWithCaveats(i, entries[i]);
			}
		}			
	};
	
	Promise<void> mergeGeometries(Geometry::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Mat4d transform, GeometryAccumulator& output);
	Promise<void> mergeGeometries(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Mat4d transform, GeometryAccumulator& output);
	
	Promise<void> collectTagNames(Geometry::Reader input, kj::HashSet<kj::String>& output);
	Promise<void> collectTagNames(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& output);
};

inline Vec3u locationInGrid(Vec3d point, Vec3d min, Vec3d max, Vec3u size) {
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

inline Vec3u locationInGrid(Vec3d point, const cu::CartesianGrid grid) {
	Vec3d min { grid.getXMin(), grid.getYMin(), grid.getZMin() };
	Vec3d max { grid.getXMax(), grid.getYMax(), grid.getZMax() };
	Vec3u size { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	return locationInGrid(point, min, max, size);
}

Vec3u locationInGrid(Vec3d point, CartesianGrid::Reader reader);

inline double rayCastTriangle(const Vec3d point, const Vec3d direction, const Vec3d triangle[3]) {
	using Eigen::seq;
	using Eigen::placeholders::all;
	
	Mat3d m;
	m(0, all) = direction;
	m(1, all) = triangle[1] - triangle[0];
	m(2, all) = triangle[2] - triangle[0];
	
	Vec3d v = point - triangle[0];
	Vec3d vi = m.partialPivLu().solve(v);
	
	double l = -v(0);
	double inf = std::numeric_limits<double>::infinity();
	
	if(l < 0)
		return inf;
	
	if(vi(1) < 0 || vi(2) < 0 || vi(1) + vi(2) > 1)
		return inf;
	
	return l;
}

struct IntersectResult {
	double l;
	size_t iMesh;
	size_t iElement;
};

inline IntersectResult intersectGeometry(const Vec3d p1, const Vec3d p2, const cu::MergedGeometry geometry, const cu::IndexedGeometry index) {
	constexpr double inf = std::numeric_limits<double>::infinity();
	double l = inf;

	const auto grid = index.getGrid();
	Vec3u i1 = locationInGrid(p1, grid);
	Vec3u i2 = locationInGrid(p2, grid);
	
	Vec3u gridSize { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	Vec3u imin = i1.array().min(i2.array());
	Vec3u imax = i1.array().max(i2.array());
	
	const auto indexData = index.getData().getData();
	
	size_t iMesh = 0;
	size_t iElement = 0;
	
	for(size_t iX = imin[0]; iX <= imax[0]; ++iX) {
	for(size_t iY = imin[1]; iY <= imax[1]; ++iY) {
	for(size_t iZ = imin[2]; iZ <= imax[2]; ++iZ) {
		size_t globalIdx = (iX * gridSize[1] + iY) * gridSize[2] + iZ;
		
		const auto indexNode = indexData[globalIdx];
		for(size_t iRef = 0; iRef < indexNode.size(); ++iRef) {
			auto elementRef = indexNode[iRef];
			
			auto refMeshIdx = elementRef.getMeshIndex();
			auto refElementIdx = elementRef.getElementIndex();
			
			// Load referenced mesh
			auto refMesh = geometry.getEntries()[refMeshIdx].getMesh();
			auto meshIndices = refMesh.getIndices();
			auto meshVertices = refMesh.getVertices().getData();
			
			// How to treat each element depends on the mesh type
			if(refMesh.hasTriMesh()) {
				Vec3d triangle[3];
				
				for(size_t i = 0; i < 3; ++i) {
					auto pointIdx = meshIndices[3 * refElementIdx + i];
					
					for(size_t j = 0; j < 3; ++j)
						triangle[i][j] = meshVertices[3 * pointIdx + j];
				}
				
				double lCast = rayCastTriangle(p1, p2 - p1, triangle);
				if(lCast < l) {
					l = lCast;
					iMesh = refMeshIdx;
					iElement = refElementIdx;
				}
			} else if(refMesh.hasPolyMesh()) {
				auto polyList = refMesh.getPolyMesh();
				
				Vec3d triangle[3];
				
				auto polyStart = polyList[refElementIdx];
				auto polyEnd   = polyList[refElementIdx + 1];
				
				auto polyStartIdx = meshIndices[polyStart];
				for(size_t i = 0; i < 3; ++i) {
					triangle[0][i] = meshVertices[3 * polyStartIdx + i];
				}
				
				for(size_t iEl = polyStart + 1; iEl < polyEnd - 1; ++iEl) {
					for(size_t dEl = 0; dEl < 2; ++dEl) {
						auto pointIdx = meshIndices[iEl + dEl];
						
						for(size_t i = 0; i < 3; ++i) {
							triangle[dEl + 1][i] = meshVertices[3 * pointIdx + i];
						}
					}
				}
				
				double lCast = rayCastTriangle(p1, p2 - p1, triangle);
				if(lCast < l) {
					l = lCast;
					iMesh = refMeshIdx;
					iElement = refElementIdx;
				}
			}
		}
	}}}
	
	IntersectResult result;
	result.l = l;
	result.iMesh = iMesh;
	result.iElement = iElement;
	
	return result;
}

}