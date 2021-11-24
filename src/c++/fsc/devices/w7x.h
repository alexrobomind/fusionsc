#include <fsc/magnetics.capnp.h>
#include <fsc/devices/w7x.capnp.h>
#include <fsc/offline.capnp.h>

#include <kj/map.h>

#include "../common.h"
#include "../magnetics.h"
#include "../data.h"


namespace fsc {
	
namespace devices { namespace w7x {

/**
 * Magnetic field resolver that processes W7-X configuration descriptions and
 * CoilsDB references.
 */
class CoilsDBResolver : public FieldResolverBase {	
	constexpr static unsigned int N_MAIN_COILS = 7;
	constexpr static unsigned int N_MODULES = 10;
	constexpr static unsigned int N_TRIM_COILS = 5;
	constexpr static unsigned int N_CONTROL_COILS = 10;
	
	constexpr static uint32_t MAIN_COIL_WINDINGS = 108;
	
	CoilsDBResolver(LibraryThread& lt, CoilsDB::Client backend);
	
	kj::TreeMap<uint64_t, DataRef<Filament>::Client> coils;
	kj::TreeMap<ID, LocalDataRef<CoilFields>> coilPacks;
	
	CoilsDB::Client backend;
	
	Promise<void> processField   (MagneticField::Reader input, MagneticField::Builder output, ResolveContext context) override;
	Promise<void> processFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveContext context) override;
	
	// Extra function to handle the "coilsAndCurrents" type of W7-X magnetic config
	Promise<void> coilsAndCurrents(MagneticField::W7xMagneticConfig::CoilsAndCurrents::Reader reader, MagneticField::Builder output, ResolveContext context);
	
	// Returns a set of magnetic fields required for processing a configuration node.
	// Caches the field set
	LocalDataRef<CoilFields> getCoilFields(W7XCoilSet::Reader reader);
	void buildCoilFields(W7XCoilSet::Reader reader, CoilFields::Builder coilPack);
	
	// Loads the coils from the backend coilsDB. Caches the result
	DataRef<Filament>::Client getCoil(uint64_t cdbID);	
};

/**
 * Constructs a new client for the W7-X coils DB that connects to the webservice on the given address.
 */
CoilsDB::Client newCoilsDBFromWebservice(Promise<Own<kj::NetworkAddress>> address, LibraryThread& lt);

/**
 * Constructs a new client that uses an offline data source. Can take a fallback client that will be
 * used when the requested data are not available.
 */
CoilsDB::Client newCoilsDBFromOfflineData(
	LocalDataRef<OfflineData> offlineData,
	LibraryThread& lt,
	CoilsDB::Client passthrough
		= capnp::Capability::Client(KJ_EXCEPTION(FAILED, "Data not available and no passthrough specified")).castAs<CoilsDB>()
);

/*
// NOTE: This will be implemented as soon as the data archiving feature is implemented.

// CoilsDB implementation that references a mapped archive file.
class CoilsDBArchive : public CoilsDB::Server {
	
};*/

}}

}