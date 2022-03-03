#pragma once

#include "local.h"
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
	Promise<void> merge(MergeContext context) override;
	Promise<void> index(IndexContext context) override;
};

}