#include <fsc/magnetics.capnp.h>
#include <fsc/geometry.capnp.h>

namespace fsc {

namespace devices { namespace jtext {
	
kj::StringPtr exampleGeqdsk();

ToroidalGrid::Reader defaultGrid();
CartesianGrid::Reader defaultGeometryGrid();

FieldResolver::Client newFieldResolver();
GeometryResolver::Client newGeometryResolver();

}}

}