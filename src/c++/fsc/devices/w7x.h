#include <fsc/magnetics.capnp.h>
#include <fsc/devices/w7x.capnp.h>
#include <fsc/offline.capnp.h>

#include <kj/map.h>

#include "../common.h"
#include "../magnetics.h"
#include "../geometry.h"
#include "../data.h"


namespace fsc {
	
namespace devices { namespace w7x {
	
static constexpr kj::StringPtr DEFAULT_COILSDB = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest"_kj;
static constexpr kj::StringPtr DEFAULT_COMPONENTSDB = "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest"_kj;

/**
 * Constructs a new client for the W7-X coils DB that connects to the webservice on the given address.
 */
CoilsDB::Client newCoilsDBFromWebservice(kj::StringPtr addressPrefix);
ComponentsDB::Client newComponentsDBFromWebservice(kj::StringPtr address);

FieldResolver::Client newW7xFieldResolver();
FieldResolver::Client newCoilsDBResolver(CoilsDB::Client coilsDB);
FieldResolver::Client newConfigDBResolver(CoilsDB::Client coilsDB);

GeometryResolver::Client newW7xGeometryResolver();
GeometryResolver::Client newComponentsDBResolver(ComponentsDB::Client componentsDB);

Provider::Client newProvider();

void buildCoilFields(W7XCoilSet::Reader in, W7XCoilSet::Fields::Builder out);


}}

}
