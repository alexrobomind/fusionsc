#pragma once

#include <fsc/flt.capnp.cu.h>

#include "interpolation.h"
#include "index.h"

namespace fsc {

inline EIGEN_DEVICE_FUNC Vec<double, 2> mappedPosition(const cu::FieldlineMapping::MappingFilament filament, double phi, const Vec<double, 2>& rz) {
	using Strategy = C1CubicInterpolation<double>;
	using Interpolator = NDInterpolator<1, Strategy>;
	
	double phiMin = filament.getPhiMin();
	Interpolator::Axis ax1(0, fmod(filament.getPhiMax() + 2 * pi - phiMin, 2 * pi), filament.getNPhi());
	Interpolator interp(Strategy(), {ax1});
	
	double dPhi = phi - phiMin;
	dPhi += 2 * pi;
	dPhi = fmod(dPhi, 2 * pi);
	auto data = filament.getData();
	
	Vec2d rzFilament;
	for(int dim = 0; dim < 2; ++dim) {
		auto filamentPos = [&](int i) {
			i += 1;
			return data[6 * i + dim];
		};
		rzFilament(dim) = interp(filamentPos, Vec1d {dPhi});
	}
	
	Mat2d jacobian;
	for(int idx = 0; idx < 4; ++idx) {
		auto filamentJacobian = [&](int i) {
			i += 1;
			return data[6 * i + idx + 2];
		};
		jacobian.data()[idx] = interp(filamentJacobian, Vec1d {dPhi});
	}
	
	return jacobian.inverse() * (rz - rzFilament);
}

inline EIGEN_DEVICE_FUNC Vec<double, 2> unmappedPosition(const cu::FieldlineMapping::MappingFilament filament, double phi, const Vec<double, 2>& uv) {
	using Strategy = C1CubicInterpolation<double>;
	using Interpolator = NDInterpolator<1, Strategy>;
	
	double phiMin = filament.getPhiMin();
	Interpolator::Axis ax1(0, fmod(filament.getPhiMax() + 2 * pi - phiMin, 2 * pi), filament.getNPhi());
	Interpolator interp(Strategy(), {ax1});
	
	double dPhi = phi - phiMin;
	dPhi += 2 * pi;
	dPhi = fmod(dPhi, 2 * pi);
	auto data = filament.getData();
	
	Vec2d rzFilament;
	for(int dim = 0; dim < 2; ++dim) {
		auto filamentPos = [&](int i) {
			i += 1;
			return data[6 * i + dim];
		};
		rzFilament(dim) = interp(filamentPos, Vec1d {dPhi});
	}
	
	Mat2d jacobian;
	for(int idx = 0; idx < 4; ++idx) {
		auto filamentJacobian = [&](int i) {
			i += 1;
			return data[6 * i + idx + 2];
		};
		jacobian.data()[idx] = interp(filamentJacobian, Vec1d {dPhi});
	}
	
	return rzFilament + jacobian * uv;
}

inline EIGEN_DEVICE_FUNC cu::FieldlineMapping::MappingFilament findNearestFilament(const cu::FieldlineMapping mapping, const Vec<double, 3>& xyz) {
	KDTreeIndex<3> index(mapping.getIndex());
	
	auto findResult = index.findNearest(xyz);
	return mapping.getFilaments()[findResult.key];
}

}