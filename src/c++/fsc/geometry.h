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
	auto fraction = (point - min) / (max - min);
	auto perCell = fraction * size;
	Vec3i result = perCell;
	
	for(size_t i = 0; i < 3; ++i) {
		if(result[i] < 0)
			result[i] = 0;
		
		if(result[i] >= size[i])
			result[i] = size[i] - 1;
	}
	
	return result;
}

inline Vec3u locationInGrid(Vec3d point, const CupnpVal<GartesianGrid> grid) {
	Vec3d min { grid.getXMin(), grid.getYMin(), grid.getZMin() };
	Vec3d max { grid.getXMax(), grid.getYMax(), grid.getZMax() };
	Vec3u size { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	return locationInGrid(point, min, max, size);
}

Vec3u locationInGrid(Vec3d point, GartesianGrid::Reader reader);

}