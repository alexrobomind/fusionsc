#pragma once

#include <fsc/magnetics.capnp.h>
#include <kj/map.h>

#include "common.h"
#include "local.h"
#include "data.h"
#include "tensor.h"

#include "magnetics-local.h"

namespace fsc {

ToroidalGridStruct readGrid(ToroidalGrid::Reader in, unsigned int maxOrdinal);
void writeGrid(const ToroidalGridStruct& grid, ToroidalGrid::Builder out);

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
FieldCalculator::Client newCPUFieldCalculator(LibraryThread& lt);

#ifdef FSC_WITH_CUDA

/**
 * Creates a new gpu-based field calculator.
 */
FieldCalculator::Client newGPUFieldCalculator(LibraryThread& lt);

#endif

// Inline Implementation

}