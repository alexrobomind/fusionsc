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

FSC_DECLARE_KERNEL(addFieldInterpKernel, FieldValuesRef, FieldValuesRef, FieldRef, ToroidalGridStruct, double);
FSC_DECLARE_KERNEL(biotSavartKernel, FieldValuesRef, FilamentRef, double, double, double, FieldValuesRef);
FSC_DECLARE_KERNEL(eqFieldKernel, FieldValuesRef, cu::AxisymmetricEquilibrium::Reader, double, FieldValuesRef);

/**
 \ingroup kernels
 */
/*EIGEN_DEVICE_FUNC inline void addFieldKernel(unsigned int idx, FieldRef out, FieldRef in, double scale) {
	out.data()[idx] += in.data()[idx] * scale;
}*/

EIGEN_DEVICE_FUNC inline void addFieldInterpKernel(unsigned int idx, FieldValuesRef out, FieldValuesRef pointsOut, FieldRef in, ToroidalGridStruct gridIn, double scale) {	
	double x = pointsOut(idx, 0);
	double y = pointsOut(idx, 1);
	double z = pointsOut(idx, 2);
	
	double r = std::sqrt(x*x + y*y);
	double phi = std::atan2(y, x);
	
	// Custom addition here
	using InterpolationStrategy = C1CubicInterpolation<double>;
	SlabFieldInterpolator<InterpolationStrategy> interpolator(InterpolationStrategy(), gridIn);
	Vec3d field = interpolator(in, Vec3d(x, y, z));
	
	double bTor = std::cos(phi) * field[1] - std::sin(phi) * field[0];
	
	out(idx, 0) += scale * field[0];
	out(idx, 1) += scale * field[1];
	out(idx, 2) += scale * field[2];
}
	
/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void biotSavartKernel(unsigned int idx, FieldValuesRef pointsOut, FilamentRef filament, double current, double coilWidth, double stepSize, FieldValuesRef out) {	
	Vec3d x_grid(pointsOut(idx, 0), pointsOut(idx, 1), pointsOut(idx, 2));
	Vec3d field_cartesian { 0, 0, 0 };
		
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
	
	out(idx, 0) += current * field_cartesian(0);
	out(idx, 1) += current * field_cartesian(1);
	out(idx, 2) += current * field_cartesian(2);
}

EIGEN_DEVICE_FUNC inline void dipoleFieldKernel(unsigned int idx, FieldValuesRef PointsOut, FieldValuesRef dipolePoints, FieldValues dipoleMoments, double dipoleRadius, double scale, FieldValuesRef out) {
	// Based on field of magnetized sphere
	// https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node61.html
	
	Vec3d field(0, 0, 0);
	Vec3D x(pointsOut(idx, 0), pointsOut(idx, 1), pointsOut(idx, 2));
	
	auto nPoints = dipoleData.dimension(0);
	for(int64_t iPoint = 0; iPoint < nPoints; ++iPoint) {
		Vec3d p(dipolePoints(iPoint, 0), dipolePoints(iPoint, 1), dipolePoints(iPoint, 2));
		Vec3d m(dipoleMoments(iPoint, 0), dipoleMoments(iPoint, 1), dipoleMoments(iPoint, 2));
		
		Vec3d r = x - p;
		double rAbs = r.norm();
		
		constexpr double mu0over4pi = 1e-7;
		constexpr double mu0 = mu0over4pi * 4 * fsc::pi;
		
		if(rAbs < dipoleRadius) {
			double volume = 4.0 / 3.0 * fsc::pi * dipoleRadius * dipoleRadius * dipoleRadius;
			Vec3d magnetization = m / volume;
			
			field += 2.0 / 3.0 * mu0 * magnetization;
			continue;
		}
		
		Vec3d rNorm = r / rAbs;
		
		Vec3d unscaled = 3 * rNorm * (rNorm.dot(m)) - m;
		field += scale * mu0over4pi / (rAbs * rAbs * rAbs) * unscaled;
	}
	
	out(idx, 0) += field(0);
	out(idx, 1) += field(1);
	out(idx, 2) += field(2);
}

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void eqFieldKernel(unsigned int idx, FieldValuesRef pointsOut, fsc::cu::AxisymmetricEquilibrium::Reader equilibrium, double scale, FieldValuesRef out) {	
	double x = pointsOut(idx, 0);
	double y = pointsOut(idx, 1);
	double z = pointsOut(idx, 2);
	
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
		return psiIn.getData()[idx];
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
		
		Interpolator interp(DiffStrategy(), { Axis(equilibrium.getZMin(), equilibrium.getZMax(), psiNZ), Axis(equilibrium.getRMin(), equilibrium.getRMax(), psiNR) });
		
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
		
		Interpolator interp(InterpStrategy(), { Axis(equilibrium.getFluxAxis(), equilibrium.getFluxBoundary(), nPsiBtor) });
		
		bTor = interp(bTorNorm, Vec1d { psiVal }) / r;
	}
	
	double bR = -dPsi_dZ / r;
	double bZ =  dPsi_dR / r;
	
	double bX = (bR * x - bTor * y) / r;
	double bY = (bR * y + bTor * x) / r;
	
	out(idx, 0) += scale * bX;
	out(idx, 1) += scale * bY;
	out(idx, 2) += scale * bZ;
}

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void surfaceFourier(unsigned int idx, fsc::cu::FourierKernelData::Builder data, FieldRef fieldData) {
}

}}