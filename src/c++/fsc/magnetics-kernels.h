#pragma once

#include "magnetics-local.h"

namespace fsc { namespace internal {

using Field = Eigen::Tensor<double, 4>;
using FieldRef = Eigen::TensorMap<Field>;

using MFilament = Eigen::Tensor<double, 2>;
using FilamentRef = Eigen::TensorMap<MFilament>;
	
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

	Vec3d field_cartesian;
	for(unsigned int i = 0; i < 3; ++i)
		field_cartesian(i) = 0;
	
	// bool report = (idx % 53 == 0);
	// bool report = (idx == 639);
	
	//if(report)
	//	KJ_LOG(WARNING, idx, i_phi, i_z, i_r);
	
	auto n_points = filament.dimension(1);	
	for(int i_fil = 0; i_fil < n_points - 1; ++i_fil) {
		// Extract current filament
		Vec3d x1 = filament.chip(i_fil, 1);
		Vec3d x2 = filament.chip(i_fil + 1, 1);
		
		// Calculate step and no. of steps
		auto dxtot = (x2 - x1).eval();
		double dxnorm = norm((Vec3d) dxtot);
		int n_steps = (int) (dxnorm / stepSize + 1);
		Vec3d dx = dxtot * (1.0 / n_steps);
		
		for(int i_step = 0; i_step < n_steps; ++i_step) {
			auto x = x1 + ((double) i_step + 0.5) * dx;
			
			Vec3d dr = (x_grid - x).eval();
						
			auto distance = dr.square().sum().sqrt();
			auto useDistance = distance.cwiseMax(distance.constant(coilWidth)).eval();
			TensorFixedSize<double, Eigen::Sizes<>> dPow3 = useDistance * useDistance * useDistance;
			
			constexpr double mu0over4pi = 1e-7;
			field_cartesian += mu0over4pi * cross(dx, dr) / dPow3();
		}
	}
	
	/*if(report) {
		Tensor<double, 0> dotProt = (x_grid * field_cartesian).sum();
		KJ_LOG(WARNING, "Biot savart result");
		KJ_LOG(WARNING, x_grid(0), x_grid(1), x_grid(2));
		KJ_LOG(WARNING, field_cartesian(0), field_cartesian(1), field_cartesian(2));
		KJ_LOG(WARNING, dotProt());
	}*/
	
	double phi = grid.phi(i_phi);
	double fieldR   = field_cartesian(0) * cos(phi) + field_cartesian(1) * sin(phi);
	double fieldZ   = field_cartesian(2);
	double fieldPhi = field_cartesian(1) * cos(phi) - field_cartesian(0) * sin(phi);
	
	/*if(report) {
		KJ_LOG(WARNING, fieldR, fieldZ, fieldPhi);
		KJ_LOG(WARNING, fieldPhi * grid.r(i_r));
		KJ_LOG(WARNING, coilWidth, stepSize, current);
	}*/
	
	double* outData = out.data();
	outData[3 * idx + 0] += current * fieldPhi;
	outData[3 * idx + 1] += current * fieldZ;
	outData[3 * idx + 2] += current * fieldR;
}

}}