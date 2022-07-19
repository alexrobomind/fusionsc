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

CoilsDB::Client newCoilsDBFromOfflineData(DataRef<OfflineData>::Client offlineData);
ComponentsDB::Client newComponentsDBFromOfflineData(DataRef<OfflineData>::Client offlineData);

FieldResolver::Client newCoilsDBResolver(CoilsDB::Client coilsDB);
GeometryResolver::Client newComponentsDBResolver(ComponentsDB::Client componentsDB);

kj::Array<Temporary<MagneticField>> preheatFields(W7XCoilSet::Reader coils);

/*
// NOTE: This will be implemented as soon as the data archiving feature is implemented.

// CoilsDB implementation that references a mapped archive file.
class CoilsDBArchive : public CoilsDB::Server {
	
};*/

}}

}
