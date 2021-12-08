#pragma once

#include <fsc/geometry.capnp.h>

namespace fsc {
	
struct GeometryResolverBase : public GeometryResolver::Server {
public:
	LibraryThread lt;
	GeometryResolverBase(LibraryThread& lt);
	
	virtual Promise<void> resolve(ResolveContext context) override;
	
	virtual Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveContext context);
	        Promise<void> processTransform(Transformed<Geometry>::Reader input, Transformed<Geometry>::Builder output, ResolveContext context);
};

}