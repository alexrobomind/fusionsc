#include <fsc/magnetics.capnp.h>
#include <kj/map.h>

#include "common.h"
#include "local.h"

namespace fsc {

/**
 * Helper macro that allows you to specify the maximum version of a toroidal
 * grid which you understand. All code handling computed fields (both creating
 * and consuming) should use this macro to ensure it fully understands the
 * meaning of the grid definition. Note that fields set to default values
 * do not affect the version computation.
 */
#define FSC_TGRID_VERSION(version, grid) \
	KJ_REQUIRE( \
		::fsc::toroidalGridVersion((grid)) <= ::fsc::ToroidalGridVersion::(version), \
		"Grid too new for me to understand" \
	)

enum class ToroidalGridVersion {
	V1 = 0,
	UNKNOWN = 999 // Increase this if we ever hit version 999
};

/**
 * Computes the minimum toroidal grid version required to represent
 * this grid.
 */
ToroidalGridVersion toroidalGridVersion(ToroidalGrid::Reader grid);

class FieldResolverBase : public FieldResolver::Server {
public:
	LibraryThread lt;
	FieldResolverBase(LibraryThread& lt);
	
	virtual Promise<void> resolve(ResolveContext context) override;
	
	virtual Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveContext context);
	virtual Promise<void> processFilament(Filament::Reader input, Filament::Builder output, ResolveContext context);
};

/**
 * Creates a new field calculator.
 */
FieldCalculator::Client newFieldCalculator(LibraryThread& lt);

}