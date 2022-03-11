#pragma once

#include "local.h"
#include "data.h"
#include "tensor.h"

#include <fsc/geometry.capnp.h>

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

}