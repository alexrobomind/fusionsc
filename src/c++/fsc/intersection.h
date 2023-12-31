#pragma once

#include "geometry.h"

#include <fsc/geometry.capnp.h>
#include <fsc/geometry.capnp.cu.h>
#include <fsc/flt.capnp.cu.h>

#include <cupnp/cupnp.h>

namespace fsc {

inline EIGEN_DEVICE_FUNC Vec3u locationInGrid(Vec3d point, const cu::CartesianGrid::Reader& grid) {
	Vec3d min { grid.getXMin(), grid.getYMin(), grid.getZMin() };
	Vec3d max { grid.getXMax(), grid.getYMax(), grid.getZMax() };
	Vec3u size { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	return locationInGrid(point, min, max, size);
}

inline EIGEN_DEVICE_FUNC double vecdet(const Vec3d v0, const Vec3d v1, const Vec3d v2) {
	return
		  v0[0] * v1[1] * v2[2]
		+ v0[1] * v1[2] * v2[0]
		+ v0[2] * v1[0] * v2[1]
		- v0[2] * v1[1] * v2[0]
		- v0[0] * v1[2] * v2[1]
		- v0[1] * v1[0] * v2[2]
	;
}

inline EIGEN_DEVICE_FUNC double rayCastTriangle(const Vec3d point, const Vec3d direction, const Vec3d triangle[3]) {
	using Eigen::seq;
	using Eigen::all;
	
	Vec3d v = point - triangle[0];
	
	#ifdef CUPNP_DEVICE_COMPILATION_PHASE
		Vec3d v1 = direction;
		Vec3d v2 = triangle[1] - triangle[0];
		Vec3d v3 = triangle[2] - triangle[0];
		
		// Solve the system [v1, v2, v3] vi = v via Cramer's rule
		double invDet = 1 / vecdet(v1, v2, v3);
		Vec3d vi(
			vecdet(v, v2, v3) * invDet,
			vecdet(v1, v, v3) * invDet,
			vecdet(v1, v2, v) * invDet
		);
	#else
		Mat3d m;
		m(all, 0) = direction;
		m(all, 1) = triangle[1] - triangle[0];
		m(all, 2) = triangle[2] - triangle[0];
		
		Vec3d vi = m.partialPivLu().solve(v);
	#endif
	
	/* Mat3d m;
	m(all, 0) = direction;
	m(all, 1) = triangle[1] - triangle[0];
	m(all, 2) = triangle[2] - triangle[0]; 
	
	Vec3d vi = m.partialPivLu().solve(v);*/
	
	double l = -vi(0);
	double inf = std::numeric_limits<double>::infinity();
	
	if(l < 0)
		return inf;
	
	if(vi(1) < 0 || vi(2) < 0 || vi(1) + vi(2) > 1)
		return inf;
	
	/*Vec3d e1(1.0, 0.0, 0.0);
	Vec3d e2(0.0, 1.0, 0.0);
	Vec3d e3(0.0, 0.0, 1.0);
	
	Vec3d c1 = point + m * (e1 * l - e2 * (vi[1] - 1) - e3 * vi[2]) - triangle[1];
	Vec3d c2 = point + m * (e1 * l - e2 * vi[1] - e3 * (vi[2] - 1)) - triangle[2];
	Vec3d c3 = point + m * (e1 * l - e2 * vi[1] - e3 * vi[2]) - triangle[0];
	Vec3d c4 = m * e1 - direction;
	
	KJ_DBG(c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2], c4[0], c4[1], c4[2]);*/
	
	return l;
}

/**
 * \ingroup geometry
 */
struct IntersectResult {
	double l;
	size_t iMesh;
	size_t iElement;
};


/**
 * \ingroup geometry
 * \return The new number of events in the event buffer, or eventBuffer.size() to indicate that we ran out of space.
 */
inline EIGEN_DEVICE_FUNC uint32_t intersectGeometryAllEvents(
	const Vec3d p1, const Vec3d p2,
	cu::MergedGeometry::Reader geometry, cu::IndexedGeometry::Reader index, cu::IndexedGeometry::IndexData::Reader indexData,
	
	double lMax,
	
	cupnp::List<cu::FLTKernelEvent>::Builder eventBuffer, uint32_t eventCount
) {	
	double distanceP1P2 = (p2 - p1).norm();
	Vec3d dp = (p2 - p1);

	const auto grid = index.getGrid();
	Vec3u i1 = locationInGrid(p1, grid);
	Vec3u i2 = locationInGrid(p2, grid);
	
	Vec3u gridSize { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	Vec3u imin = i1.array().min(i2.array());
	Vec3u imax = i1.array().max(i2.array());
	
	const auto indexGridData = indexData.getGridContents().getData();
	
	size_t iMesh = 0;
	size_t iElement = 0;
		
	auto currentEvent = [&]() {
		return eventBuffer[eventCount];
	};
	
	// Note: The following parts contain an out-of-band return. Therefore,
	// they need to be implemented as macros.
	
	#define FSC_NEXT_EVENT() { \
		if(eventCount >= eventBuffer.size() - 1) \
			return eventBuffer.size(); \
		\
		++eventCount; \
	}
	
	#define FSC_HANDLE_TRIANGLE(p1, p2, triangle, meshIdx, elementIdx) { \
		double lCast = rayCastTriangle(p1, p2 - p1, triangle); \
		\
		if(lCast < lMax) { \
			auto event = currentEvent(); \
			event.setDistance(distanceP1P2 * lCast); \
			event.setX(p1[0] + lCast * dp[0]); \
			event.setY(p1[1] + lCast * dp[1]); \
			event.setZ(p1[2] + lCast * dp[2]); \
			\
			auto geoHit = event.mutateGeometryHit(); \
			geoHit.setMeshIndex(meshIdx); \
			geoHit.setElementIndex(elementIdx); \
			\
			FSC_NEXT_EVENT() \
		} \
	}
	
	for(size_t iX = imin[0]; iX <= imax[0]; ++iX) {
	for(size_t iY = imin[1]; iY <= imax[1]; ++iY) {
	for(size_t iZ = imin[2]; iZ <= imax[2]; ++iZ) {
		size_t globalIdx = (iX * gridSize[1] + iY) * gridSize[2] + iZ;
		
		const auto indexNode = indexGridData[globalIdx];
		
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
				
				FSC_HANDLE_TRIANGLE(p1, p2, triangle, refMeshIdx, refElementIdx);
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
				
				FSC_HANDLE_TRIANGLE(p1, p2, triangle, refMeshIdx, refElementIdx);
			}
		}
	}}}
	
	return eventCount;
	
	#undef FSC_NEXT_EVENT
	#undef FSC_HANDLE_TRIANGLE
}



/**
 * \ingroup geometry
 */
inline EIGEN_DEVICE_FUNC IntersectResult intersectGeometryFirstHit(
	const Vec3d p1, const Vec3d p2,
	cu::MergedGeometry::Reader geometry, cu::IndexedGeometry::Reader index, cu::IndexedGeometry::IndexData::Reader indexData
) {
	constexpr double inf = std::numeric_limits<double>::infinity();
	double l = inf;

	const auto grid = index.getGrid();
	Vec3u i1 = locationInGrid(p1, grid);
	Vec3u i2 = locationInGrid(p2, grid);
	
	Vec3u gridSize { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	Vec3u imin = i1.array().min(i2.array());
	Vec3u imax = i1.array().max(i2.array());
	
	const auto indexGridData = indexData.getGridContents().getData();
	
	size_t iMesh = 0;
	size_t iElement = 0;
	
	for(size_t iX = imin[0]; iX <= imax[0]; ++iX) {
	for(size_t iY = imin[1]; iY <= imax[1]; ++iY) {
	for(size_t iZ = imin[2]; iZ <= imax[2]; ++iZ) {
		size_t globalIdx = (iX * gridSize[1] + iY) * gridSize[2] + iZ;
		
		const auto indexNode = indexGridData[globalIdx];
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