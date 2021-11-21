#include <fsc/magnetics.capnp.h>
#include <fsc/devices/w7x.capnp.h>

#include <kj/map.h>

namespace fsc {
	
namespace devices { namespace w7x {

/**
 * Magnetic field resolver that processes W7-X configuration descriptions and
 * CoilsDB references.
 */
class CoilsDBResolver : public FieldResolverBase {	
	constexpr size_t N_MAIN_COILS = 7;
	constexpr size_t N_MODULES = 10;
	constexpr size_t N_TRIM_COILS = 5;
	constexpr size_t N_CONTROL_COILS = 10;
	
	constexpr uint32_t MAIN_COIL_WINDINGS = 108;
	
	kj::TreeMap<uint64_t, DataRef<Filament>::Client> coils;
	kj::TreeMap<ID, LocalDataRef<CoilFields>> coilPacks;
	
	fsc::w7x::CoilsDB backend;
	
	Promise<void> customField   (MagneticField::Reader input, MagneticField::Builder output, ResolveContext& context) override;
	Promise<void> customFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveContext& context) override;
	
	// Returns a set of magnetic fields required for processing a configuration node.
	LocalDataRef<CoilFields> getCoilFields(W7XCoilSet::Reader reader);
	
	DataRef<Filament>::Client getCoil(uint64_t cdbID);
	
	
	void buildCoilFields(W7XCoilSet::Reader reader, CoilFields::Builder coilPack);
};

// CoilsDB implementation that downloads the coil from a remote location
// Does not perform any local caching.
class CoilsDBWebserviceClient : public CoilsDB::Server {
	
};

/*
// NOTE: This will be implemented as soon as the data archiving feature is implemented.

// CoilsDB implementation that references a mapped archive file.
class CoilsDBArchive : public CoilsDB::Server {
	
};*/

}}

}