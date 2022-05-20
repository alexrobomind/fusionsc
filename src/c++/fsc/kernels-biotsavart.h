#pragma once

#include "vector.h"
#include "tensor.h"
#include "grids.h"
#include "kernels.h"

#include <utility>
#include <kj/function.h>

namespace fsc { namespace internal {

using Field = Eigen::Tensor<double, 4>;
using FieldRef = Eigen::TensorMap<Field>;

using MFilament = Eigen::Tensor<double, 2>;
using FilamentRef = Eigen::TensorMap<MFilament>;

template<typename Device>
void addFields(Device& device, FieldRef field1, FieldRef field2, double scale, kj::Function<void()>&& done) {
	field1.device(device, mv(done)) = field1 + field2 * scale;
	// potentiallySynchronize(device);
}

#ifdef FSC_WITH_CUDA
#ifndef EIGEN_GPUCC

// addFields for GPU device must be instantiated in CUDA compiler
extern template void addFields<Eigen::GpuDevice>(Eigen::GpuDevice&, FieldRef, FieldRef, double, kj::Function<void()>&&);

#endif
#endif
/*
template<typename T1, typename T2>
auto cross(T1 exp1, T2 exp2, unsigned int axis) {
	auto v0 = exp1.slice(axis, 1) * exp2.slice(axis, 2) - exp2.slice(axis, 1) * exp1.slice(axis, 2);
	auto v1 = exp1.slice(axis, 2) * exp2.slice(axis, 0) - exp2.slice(axis, 2) * exp1.slice(axis, 0);
	auto v2 = exp1.slice(axis, 0) * exp2.slice(axis, 1) - exp2.slice(axis, 0) * exp1.slice(axis, 1);
	
	return v0.concatenate(v1, axis).concatenate(v2, axis);
}

template<typename Device>
inline void biotSavart(Device& device, Callback<> done, ToroidalGridStruct grid, MFilament filament, double current, double coilWidth, double stepSize, FieldRef out) {
	using i5 = Eigen::array<int, 5>;
	
	auto nR = grid.nR;
	auto nZ = grid.nZ;
	auto nPhi = grid.nPhi;
	auto nFil = filament.dimension(1) - 1;
	
	// Compute step sizes for each sub filament
	Tensor<unsigned int, 1> nSteps(nFil);
	unsigned int totalSteps = 0;
	for(size_t i = 0; i < nFil; ++i) {
		auto x1 = filament.chip(i, 1);
		auto x2 = filament.chip(i + 1, 1);
		Tensor<double, 1> norm = (x2 - x1).square().sum().sqrt();
		
		nSteps(i) = 1 + (unsigned int) (norm() / stepSize);
		totalSteps += nSteps(i);
	}
	
	// Expand coil filament
	Tensor<double, 2> points(3, nFil + 1);
	Tensor<double, 2> dx(3, nFil + 1);
	unsigned int nPoints = 0;
	for(size_t i = 0; i < nFil; ++i) {
		auto x1 = filament.chip(i, 1);
		auto x2 = filament.chip(i + 1, 1);
		unsigned int n = nSteps(iFil);
		Tensor<double, 3> dx = (x2 - x1) / n;
		
		for(unsigned int i = 0; i < n; ++i) {
			dx.chip(nPoints, 1) = dx;
			points.chip(nPoints++, 1) = x1 + dx * i;
		}
	}
	
	// Compute grid coordinates
	Tensor<double, 1> r  (nR);
	Tensor<double, 1> z  (nZ);
	Tensor<double, 1> phi(nPhi);
	
	for(unsigned int i = 0; i < nPhi; ++i)
		phi(i) = grid.phi(i);
	
	for(unsigned int i = 0; i < nR; ++i)
		r(i) = grid.r(i);
	
	for(unsigned int i = 0; i < nZ; ++i)
		z(i) = grid.z(i);
	
	MappedTensor<Tensor<double, 1>> rGpu(r);
	MappedTensor<Tensor<double, 1>> zGpu(z);
	MappedTensor<Tensor<double, 1>> phiGpu(phi);
	MappedTensor<Tensor<double, 2>> pointsGpu(points);
	MappedTensor<Tensor<double, 2>> dxGpu(dx);
	
	gGpu.updateDevice();
	zGpu.updateDevice();
	phiGpu.updateDevice();
	pointsGpu.updateDevice();
	dxGpu.updateDevice();
	
	// Broadcast into compatible shapes
	auto rg =     rGpu.reshape(i5({1, 1, nPhi, 1, 1)}).broadcast(i5({1, nPoints, 1, nZ, nR)});
	auto zg =     zGpu.reshape(i5({1, 1, 1, nZ, 1)}).broadcast(i5({1, nPoints, nPhi, 1, nR)});
	auto phig = phiGpu.reshape(i5({1, 1, 1, 1, nR)}).broadcast(i5({1, nPoints, nPhi, nZ, 1)});
	
	auto filg = pointsGpu.reshape(i5({3, nFil, 1, 1, 1)}).broadcast(i5({1, 1, nPhi, nZ, nR)});
	auto dxg  =     gxGpu.reshape(i5({3, nFil, 1, 1, 1)}).broadcast(i5({1, 1, nPhi, nZ, nR)});
	
	auto xg = rg * phig.cos();
	auto yg = rg * phig.sin();
	
	auto xyzg = xg.concatenate(yg, 0).concatenate(zg, 0);
	
	auto drg   = xyzg - filg;
	auto distance = dr.square().sum(0).sqrt();
	auto useDistance = distance.cwiseMax(coilWidth);
	
	constexpr double mu0over4pi = 1e-7;
	auto dpow3 = useDistance * useDistance * useDistance;
	
	auto bsfield = cross(dxg, drg, 0) * mu0over4pi / dpow3;
	out.device(device, done) += bsfield.sum(1);
}*/
	
	
EIGEN_DEVICE_FUNC inline void biotSavartKernel(const unsigned int idx, ToroidalGridStruct grid, FilamentRef filament, double current, double coilWidth, double stepSize, FieldRef out) {
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
	
	int i_phi = midx[0];
	int i_z   = midx[1];
	int i_r   = midx[2];

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

}}