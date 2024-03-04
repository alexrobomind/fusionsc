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

using MFilament = Eigen::Tensor<double, 2>;
using FilamentRef = Eigen::TensorMap<MFilament>;

FSC_DECLARE_KERNEL(addFieldKernel, FieldRef, FieldRef, double);
FSC_DECLARE_KERNEL(addFieldInterpKernel, FieldRef, ToroidalGridStruct, FieldRef, ToroidalGridStruct, double);
FSC_DECLARE_KERNEL(biotSavartKernel, ToroidalGridStruct, FilamentRef, double, double, double, FieldRef);
FSC_DECLARE_KERNEL(eqFieldKernel, ToroidalGridStruct, cu::AxisymmetricEquilibrium::Reader, double, FieldRef);

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void addFieldKernel(unsigned int idx, FieldRef out, FieldRef in, double scale) {
	out.data()[idx] += in.data()[idx] * scale;
}

EIGEN_DEVICE_FUNC inline void addFieldInterpKernel(unsigned int idx, FieldRef out, ToroidalGridStruct gridOut, FieldRef in, ToroidalGridStruct gridIn, double scale) {
	// Copied from biotSavartKernel
	int midx[3];
	
	{
		// Decode index using column major layout
		// in which the first index has stride 1
		unsigned int tmp = idx;
		for(int i = 0; i < 3; ++i) {
			midx[i] = tmp % out.dimension(i+1);
			tmp /= out.dimension(i+1);
		}
	}
	
	int i_r   = midx[0];
	int i_z   = midx[1];
	int i_phi = midx[2];

	Vec3d x_grid = gridIn.xyz(i_phi, i_z, i_r);
	
	// Custom addition here
	using InterpolationStrategy = C1CubicInterpolation<double>;
	SlabFieldInterpolator<InterpolationStrategy> interpolator(InterpolationStrategy(), gridIn);
	Vec3d field = interpolator.inSlabOrientation(in, x_grid);
	
	double* outData = out.data();	
	outData[3 * idx + 0] += scale * field[0];
	outData[3 * idx + 1] += scale * field[1];
	outData[3 * idx + 2] += scale * field[2];
}
	
/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void biotSavartKernel(unsigned int idx, ToroidalGridStruct grid, FilamentRef filament, double current, double coilWidth, double stepSize, FieldRef out) {
	int midx[3];
	
	{
		// Decode index using column major layout
		// in which the first index has stride 1
		unsigned int tmp = idx;
		for(int i = 0; i < 3; ++i) {
			midx[i] = tmp % out.dimension(i+1);
			tmp /= out.dimension(i+1);
		}
	}
	
	int i_r = midx[0];
	int i_z   = midx[1];
	int i_phi   = midx[2];

	Vec3d x_grid = grid.xyz(i_phi, i_z, i_r);
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
	
	double r = grid.r(i_r);
	double reference = 2e-7 / r * sin(atan2(1, r));
	
	double phi = grid.phi(i_phi);
	double fieldR   = field_cartesian(0) * cos(phi) + field_cartesian(1) * sin(phi);
	double fieldZ   = field_cartesian(2);
	double fieldPhi = field_cartesian(1) * cos(phi) - field_cartesian(0) * sin(phi);
	
	double* outData = out.data();	
	outData[3 * idx + 0] += current * fieldPhi;
	outData[3 * idx + 1] += current * fieldZ;
	outData[3 * idx + 2] += current * fieldR;
}

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void eqFieldKernel(unsigned int idx, ToroidalGridStruct grid, fsc::cu::AxisymmetricEquilibrium::Reader equilibrium, double scale, FieldRef out) {
	int midx[3];
	
	{
		// Decode index using column major layout
		// in which the first index has stride 1
		unsigned int tmp = idx;
		for(int i = 0; i < 3; ++i) {
			midx[i] = tmp % out.dimension(i+1);
			tmp /= out.dimension(i+1);
		}
	}
	
	int i_r = midx[0];
	int i_z   = midx[1];
	int i_phi   = midx[2];
	
	double z = grid.z(i_z);
	double r = grid.r(i_r);
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
	
	double* outData = out.data();	
	outData[3 * idx + 0] += scale * bTor;
	outData[3 * idx + 1] += scale * bZ;
	outData[3 * idx + 2] += scale * bR;
}

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void surfaceFourier(unsigned int idx, ToroidalGridStruct grid, TensorMap<Tensor<double, 4>> fieldData,

}}