#pragma once

#include <fsc/magnetics.capnp.cu.h>

#include "grids.h"

#include "kernels/kernels.h"

#include "vector.h"
#include "tensor.h"
#include "grids.h"
#include "interpolation.h"

namespace fsc { namespace kernels {

using Field = Eigen::Tensor<double, 4>;
using FieldRef = Eigen::TensorMap<Field>;

using FieldValues = Eigen::Tensor<double, 2>;
using FieldValuesRef = Eigen::TensorMap<FieldValues>;

using MFilament = Eigen::Tensor<double, 2>;
using FilamentRef = Eigen::TensorMap<MFilament>;

using MagTransform = Eigen::Matrix<double, 3, 4>;
	
struct MagKernelContext {
	FieldValuesRef field;
	FieldValuesRef points;
	
	double scale = 1;
	Eigen::Matrix<double, 3, 4> transform = Eigen::Matrix<double, 3, 4>::Zero();
	
	bool transformed = false;
	
	MagKernelContext(FieldValuesRef pointsIn, FieldValuesRef fieldOut) :
		points(pointsIn), field(fieldOut)
	{		
		transform(0, 0) = 1;
		transform(1, 1) = 1;
		transform(2, 2) = 1;
	}
	
	inline MagKernelContext(const MagKernelContext&) = default;
	inline MagKernelContext(MagKernelContext&&) = default;
		
	EIGEN_DEVICE_FUNC inline Vec3d getPosition(unsigned int idx) const {
		if(!transformed)
			return Vec3d(points(idx, 0), points(idx, 1), points(idx, 2));
		
		return transform * Vec4d(points(idx, 0), points(idx, 1), points(idx, 2), 1);
	}
	
	EIGEN_DEVICE_FUNC inline void addField(unsigned int idx, Vec3d fieldContrib) const {
		if(transformed) {
			Vec4d adjointTransformed = transform.transpose() * fieldContrib;
			field(idx, 0) += scale * adjointTransformed(0);
			field(idx, 1) += scale * adjointTransformed(1);
			field(idx, 2) += scale * adjointTransformed(2);
		} else {
			field(idx, 0) += scale * fieldContrib(0);
			field(idx, 1) += scale * fieldContrib(1);
			field(idx, 2) += scale * fieldContrib(2);
		}	
	}
	
	EIGEN_DEVICE_FUNC inline MagKernelContext scaleBy(double scale) const {
		MagKernelContext result = *this;
		result.scale *= scale;
		return result;
	}
};

} // namespace kernels

// We don't map contexts directly, since they don't keep the
// field alive. Instead, we map Own<...>, so that we can attach
// the storage to the mapping.
template<>
struct DeviceMapping<Own<kernels::MagKernelContext>> : public DeviceMappingBase {
	Own<kernels::MagKernelContext> pCtx;
	
	inline DeviceMapping(Own<kernels::MagKernelContext>&& p, DeviceBase& device, bool allowAlias) :
		DeviceMappingBase(device),
		pCtx(mv(p))
	{}
	
	inline void doUpdateDevice() override {}
	inline void doUpdateHost() override {}
	
	inline kernels::MagKernelContext get() { return *pCtx; }
};

namespace kernels {

FSC_DECLARE_KERNEL(addFieldInterpKernel, MagKernelContext, FieldRef, ToroidalGridStruct);
FSC_DECLARE_KERNEL(biotSavartKernel, MagKernelContext, FilamentRef, double, double);
FSC_DECLARE_KERNEL(dipoleFieldKernel, MagKernelContext, FieldValuesRef, FieldValuesRef, kj::ArrayPtr<double>, size_t, size_t);
FSC_DECLARE_KERNEL(eqFieldKernel, MagKernelContext, cu::AxisymmetricEquilibrium::Reader);

/**
 \ingroup kernels
 */
/*EIGEN_DEVICE_FUNC inline void addFieldKernel(unsigned int idx, FieldRef out, FieldRef in, double scale) {
	out.data()[idx] += in.data()[idx] * scale;
}*/

EIGEN_DEVICE_FUNC inline void addFieldInterpKernel(unsigned int idx, MagKernelContext ctx, FieldRef in, ToroidalGridStruct gridIn) {	
	using InterpolationStrategy = C1CubicInterpolation<double>;
	SlabFieldInterpolator<InterpolationStrategy> interpolator(InterpolationStrategy(), gridIn);
	
	Vec3d field = interpolator(in,  ctx.getPosition(idx));
	ctx.addField(idx, field);
}
	
/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void biotSavartKernel(unsigned int idx, MagKernelContext ctx, FilamentRef filament, double coilWidth, double stepSize) {	
	Vec3d x_grid = ctx.getPosition(idx);
	Vec3d field_cartesian { 0, 0, 0 };
	
	constexpr double current = 1;
		
	auto n_points = filament.dimension(1);
	for(int i_fil = 0; i_fil < n_points - 1; ++i_fil) {
		// Extract current filament
		Vec3d x1; Vec3d x2;
		for(unsigned int i = 0; i < 3; ++i) {
			x1[i] = filament(i, i_fil);
			x2[i] = filament(i, i_fil + 1);
		}
		
		// Calculate step and no. of steps
		auto dxtot = x2 - x1;
		double dxnorm = dxtot.norm();
		int n_steps = (int) (dxnorm / stepSize + 1);
		Vec3d dx = dxtot * (1.0 / n_steps);
		
		for(int i_step = 0; i_step < n_steps; ++i_step) {
			Vec3d x = x1 + ((double) i_step + 0.5) * dx;
			
			Vec3d dr = x_grid - x;
						
			double distance = dr.norm();			
			double useDistance = std::max(distance, coilWidth);
			double dPow3 = useDistance * useDistance * useDistance;
			
			constexpr double mu0over4pi = 1e-7;
			field_cartesian += mu0over4pi * cross(dx, dr) / dPow3;
		}
	}
	
	ctx.addField(idx, field_cartesian * current);
}

EIGEN_DEVICE_FUNC inline void dipoleFieldKernel(unsigned int idx, MagKernelContext ctx, FieldValuesRef dipolePoints, FieldValuesRef dipoleMoments, kj::ArrayPtr<double> radii, size_t start, size_t end) {
	// Based on field of magnetized sphere
	// https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node61.html
	
	Vec3d field(0, 0, 0);
	Vec3d x = ctx.getPosition(idx);
	
	auto nPoints = dipolePoints.dimension(0);
	// for(int64_t iPoint = 0; iPoint < nPoints; ++iPoint) {
	for(size_t iPoint = start; iPoint < end; ++iPoint) {
		Vec3d p(dipolePoints(iPoint, 0), dipolePoints(iPoint, 1), dipolePoints(iPoint, 2));
		Vec3d m(dipoleMoments(iPoint, 0), dipoleMoments(iPoint, 1), dipoleMoments(iPoint, 2));
		
		Vec3d r = x - p;
		double rAbs = r.norm();
		
		constexpr double mu0over4pi = 1e-7;
		constexpr double mu0 = mu0over4pi * 4 * fsc::pi;
		
		const double dipoleRadius = radii[iPoint];
		
		if(rAbs < dipoleRadius) {
			double volume = 4.0 / 3.0 * fsc::pi * dipoleRadius * dipoleRadius * dipoleRadius;
			Vec3d magnetization = m / volume;
			
			field += 2.0 / 3.0 * mu0 * magnetization;
			continue;
		}
		
		Vec3d rNorm = r / rAbs;
		
		Vec3d unscaled = 3 * rNorm * (rNorm.dot(m)) - m;
		field += mu0over4pi / (rAbs * rAbs * rAbs) * unscaled;
	}
	
	ctx.addField(idx, field);
}

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void eqFieldKernel(unsigned int idx, MagKernelContext ctx, fsc::cu::AxisymmetricEquilibrium::Reader equilibrium) {
	Vec3d xyz = ctx.getPosition(idx);
	
	double x = xyz(0);
	double y = xyz(1);
	double z = xyz(2);
	
	double r = std::sqrt(x*x + y*y);
	
	Vec2d zr(z, r);
		
	using ADS = Eigen::AutoDiffScalar<Vec2d>;
	
	using InterpStrategy = C1CubicInterpolation<double>;
	using DiffStrategy = C1CubicInterpolation<ADS>;
	
	auto psiIn = equilibrium.getPoloidalFlux();
	uint32_t psiNZ = psiIn.getShape()[0];
	uint32_t psiNR = psiIn.getShape()[1];
	
	auto bTorNormIn = equilibrium.getNormalizedToroidalField();
	auto nPsiBtor = bTorNormIn.size();
	
	auto clamp = [](int in, int end) -> unsigned int {
		if(in < 0) in = 0;
		if(in >= end) in = end - 1;
		return (unsigned int) in;
	};
	
	auto psi = [&](int iZ, int iR) -> double {	
		auto idx = psiNR * clamp(iZ, psiNZ) + clamp(iR, psiNR);
		double result = psiIn.getData()[idx];
		return result;
	};
	
	auto bTorNorm = [&](int i) -> double {		
		return bTorNormIn[clamp(i, nPsiBtor)];
	};
	
	// Compute poloidal flux and derivative
	double dPsi_dZ;
	double dPsi_dR;
	double psiVal;
	{
		using Interpolator = NDInterpolator<2, DiffStrategy>;
		using Axis = Interpolator::Axis;
		
		Interpolator interp(DiffStrategy(), { Axis(equilibrium.getZMin(), equilibrium.getZMax(), psiNZ - 1), Axis(equilibrium.getRMin(), equilibrium.getRMax(), psiNR - 1) });
		
		ADS valueAndDeriv = interp(psi, { ADS(z, 2, 0), ADS(r, 2, 1) });
		psiVal = valueAndDeriv.value();
		dPsi_dZ = valueAndDeriv.derivatives()[0];
		dPsi_dR = valueAndDeriv.derivatives()[1];
	}
	
	// Compute Bt
	double bTor;
	{
		using Interpolator = NDInterpolator<1, InterpStrategy>;
		using Axis = Interpolator::Axis;
		
		Interpolator interp(InterpStrategy(), { Axis(equilibrium.getFluxAxis(), equilibrium.getFluxBoundary(), nPsiBtor - 1) });
		
		bTor = interp(bTorNorm, Vec1d { psiVal }) / r;
	}
	
	double bR = -dPsi_dZ / r;
	double bZ =  dPsi_dR / r;
	
	double bX = (bR * x - bTor * y) / r;
	double bY = (bR * y + bTor * x) / r;
	
	ctx.addField(idx, Vec3d(bX, bY, bZ));
}

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void surfaceFourier(unsigned int idx, fsc::cu::FourierKernelData::Builder data, FieldRef fieldData) {
}

}}