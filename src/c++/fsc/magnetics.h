#pragma once

#include <fsc/magnetics.capnp.h>
#include <kj/map.h>

#include "common.h"
#include "local.h"
#include "data.h"
#include "tensor.h"

#include "grids.h"

namespace fsc {
	
/**
 * \defgroup magnetics Magnetic field specification and computation
 *
 * Magnetic field services:
 *
 * \snippet magnetics.capnp magnetics
 */

ToroidalGridStruct readGrid(ToroidalGrid::Reader in, unsigned int maxOrdinal);
void writeGrid(const ToroidalGridStruct& grid, ToroidalGrid::Builder out);

class FieldResolverBase : public FieldResolver::Server {
public:	
	virtual Promise<void> resolveField(ResolveFieldContext context) override;
	
	virtual Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context);
	virtual Promise<void> processFilament(Filament::Reader input, Filament::Builder output, ResolveFieldContext context);
};

/**
 * Creates a new field calculator.
 */
FieldCalculator::Client newFieldCalculator(Own<DeviceBase> dev);

/**
 * Creates a field resolver that will insert a cache instruction when detecting the passed field
 */
FieldResolver::Client newCache(MagneticField::Reader field, ComputedField::Reader computed);

/**
 * For testing
 */
void simpleTokamak(MagneticField::Builder output, double rMajor = 5.5, double rMinor = 1.5, unsigned int nCoils = 25, double Ip = 0.3);

// Inline Implementation

}