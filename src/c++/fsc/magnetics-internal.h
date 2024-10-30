#pragma once

#include <fsc/magnetics.capnp.h>

#include "kernels/device.h"
#include "magnetics-kernels.h"

namespace fsc { namespace internal {

// Note: It is a bit questionable to do this, but since the
// struct is only used by one TU, this should be fine.
namespace {
	// Forward declaration, only for use in magnetics-calc-field.cpp
	struct FieldCalculation;
}

struct FieldCalculatorImpl : public FieldCalculator::Server {
	using MagKernelContext = kernels::MagKernelContext;
	
	Own<DeviceBase> device;
	
	inline FieldCalculatorImpl(Own<DeviceBase> device) :
		device(mv(device))
	{}
	
	// These are implemented in magnetics.cpp
	Promise<void> evaluateXyz(EvaluateXyzContext context) override;
	Promise<void> evaluatePhizr(EvaluatePhizrContext context) override;
	Promise<void> compute(ComputeContext context) override;
	Promise<void> interpolateXyz(InterpolateXyzContext ctx) override;
	Promise<void> surfaceToMesh(SurfaceToMeshContext ctx) override;
	
	// These are implemented in magnetics-calc-fourier.cpp
	Promise<void> evalFourierSurface(EvalFourierSurfaceContext ctx) override;
	Promise<void> calculateRadialModes(CalculateRadialModesContext ctx) override;
	Promise<void> surfaceToFourier(SurfaceToFourierContext ctx) override;
	
	// These are implemented in magnetics-calc-flux.cpp
	Promise<void> calculateTotalFlux(CalculateTotalFluxContext ctx) override;

private:
	// These are implemented in magnetics-calc-field.cpp
	Promise<void> processRoot(MagneticField::Reader node, Eigen::Tensor<double, 2>&& points, Float64Tensor::Builder out);
	Promise<void> processField(FieldCalculation& calculator, MagneticField::Reader node, const MagKernelContext& ctx);
	Promise<void> processTransform(FieldCalculation& calculator, Transformed<MagneticField>::Reader node, const MagKernelContext& ctx);
	Promise<void> processFilament(FieldCalculation& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, const MagKernelContext& ctx);
};

} }
