#pragma once

#include <fsc/magnetics.capnp.h>
#include <kj/map.h>

#include "common.h"
#include "local.h"
#include "data.h"
#include "tensor.h"

#include "grids.h"

#include "kernels/device.h"

namespace fsc {
	
/**
 * \defgroup magnetics Magnetic field specification and computation
 *
 * Magnetic field services:
 *
 * \snippet magnetics.capnp magnetics
 */

bool isBuiltin(MagneticField::Reader);
bool isBuiltin(Filament::Reader);

ToroidalGridStruct readGrid(ToroidalGrid::Reader in, unsigned int maxOrdinal);
void writeGrid(const ToroidalGridStruct& grid, ToroidalGrid::Builder out);

class FieldResolverBase : public FieldResolver::Server {
public:	
	Promise<void> resolveField(ResolveFieldContext context) override;
	Promise<void> resolveFilament(ResolveFilamentContext context) override;
	
	virtual Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context);
	virtual Promise<void> processFilament(Filament::Reader input, Filament::Builder output, ResolveFieldContext context);
};

struct FieldCache {
	static kj::Array<const byte> hashPoints(Eigen::TensorMap<Eigen::Tensor<double, 2>>);
	
	virtual Maybe<Promise<LocalDataRef<Float64Tensor>>> check(kj::ArrayPtr<const byte> pointsHash, kj::ArrayPtr<const byte> fieldKey) = 0;
	virtual void put(kj::ArrayPtr<const byte> pointsHash, kj::ArrayPtr<const byte> fieldKey, Promise<LocalDataRef<Float64Tensor>>) = 0;
};

Own<FieldCache> lruFieldCache(unsigned int size);

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

/**
 *  Sets the cache key to a copy of the canonicalized field
 *
 * Returns: true if the cache key could be calculated, false if not (e.g. because of nested DataRefs)
 */
bool setCacheKey(MagneticField::Builder field);

// Inline Implementation

}