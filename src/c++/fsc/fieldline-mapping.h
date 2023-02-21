#pragma once

#include <fsc/flt.capnp.cu.h>

#include "interpolation.h"
#include "index.h"

namespace fsc {
	
Mapper::Client newMapper(FLT::Client flt, KDTreeService::Client indexer);

struct FLM {
	
	//! Computes the real-space coordinates of the last-mapped position projected to the given phi plane
	inline EIGEN_DEVICE_FUNC Vec3d unmap(double phi);
	
	//! Captures a position for mapping using the closest mapping filament
	inline EIGEN_DEVICE_FUNC void map(const Vec3d& x, bool fwd);
	
	//! Advances to a new phi position, remapping if neccessary. Returns new real-space position
	// Note: No guarantees that flm.phi == newPhi
	inline EIGEN_DEVICE_FUNC Vec3d advance(double newPhi, bool fwd);
	
	//! Unwraps the given phi value to be near the currently stored position.
	inline EIGEN_DEVICE_FUNC double unwrap(double phiIn);
	
	inline EIGEN_DEVICE_FUNC FLM(cu::FieldlineMapping mapping);
	
	cu::FieldlineMapping mapping;
	Vec2d uv;
	cu::FieldlineMapping::MappingFilament activeFilament;
	double phi;

private:
	inline EIGEN_DEVICE_FUNC void interpolate(double phi, Vec2d& rz, Mat2d& jacobian);
};

EIGEN_DEVICE_FUNC FLM::FLM(cu::FieldlineMapping mapping) :
	mapping(mapping),
	uv(0, 0),
	activeFilament(0, nullptr),
	phi(0)
{}

EIGEN_DEVICE_FUNC void FLM::interpolate(double phi, Vec2d& rz, Mat2d& jacobian) {
	using Strategy = C1CubicInterpolation<double>;
	using Interpolator = NDInterpolator<1, Strategy>;
	
	Interpolator::Axis ax1(activeFilament.getPhiStart(), activeFilament.getPhiEnd(), activeFilament.getNIntervals());
	Interpolator interp(Strategy(), {ax1});
	
	auto data = activeFilament.getData();
	
	for(int dim = 0; dim < 2; ++dim) {
		auto filamentPos = [&](int i) {
			i += 1;
			// KJ_DBG(i, 6 * i + dim, data.size());
			return data[6 * i + dim];
		};
		rz(dim) = interp(filamentPos, Vec1d {phi});
	}
	
	for(int idx = 0; idx < 4; ++idx) {
		auto filamentJacobian = [&](int i) {
			i += 1;
			// KJ_DBG(i, 6 * i + idx + 2, data.size());
			return data[6 * i + idx + 2];
		};
		jacobian.data()[idx] = interp(filamentJacobian, Vec1d {phi});
	}
}

EIGEN_DEVICE_FUNC Vec3d FLM::unmap(double phi) {
	Vec2d filRZ;
	Mat2d filJacobian;
	interpolate(phi, filRZ, filJacobian);
	
	Vec2d rz = filRZ + filJacobian * uv;
	
	return { cos(phi) * rz[0], sin(phi) * rz[0], rz[1] };
}

EIGEN_DEVICE_FUNC void FLM::map(const Vec3d& x, bool fwd) {
	double newPhi = atan2(x[1], x[0]);
	
	auto dir = fwd ? mapping.getFwd() : mapping.getBwd();
	
	// Locate nearest filament point
	KDTreeIndex<3> index(dir.getIndex());
	auto findResult = index.findNearest(x);
	
	// Decode key, high 32 bits are filament, low 32 bits are point for phi assignment
	uint32_t filamentIdx = static_cast<uint32_t>(findResult.key >> 32);
	uint32_t pointIdx    = static_cast<uint32_t>(findResult.key);
	
	activeFilament = dir.getFilaments()[filamentIdx];
	
	// Compute phi baseline
	double phiBase = activeFilament.getPhiStart() + ((activeFilament.getPhiEnd() - activeFilament.getPhiStart()) / activeFilament.getNIntervals()) * pointIdx;
	
	// Unwrap phi using phiBase
	phi = phiBase;
	phi = unwrap(newPhi);
	
	// Interpolate mapping to active plane
	Vec2d filRZ;
	Mat2d filJacobian;
	interpolate(phi, filRZ, filJacobian);
	
	// Compute mapped position
	double r = sqrt(x[0] * x[0] + x[1] * x[1]);
	double z = x[2];
	
	// KJ_DBG(r, z);
	
	uv = filJacobian.inverse() * (Vec2d { r, z } - filRZ);
}

EIGEN_DEVICE_FUNC Vec3d FLM::advance(double newPhi, bool fwd) {
	double relToRange = (newPhi - activeFilament.getPhiStart()) / (activeFilament.getPhiEnd() - activeFilament.getPhiStart()) * activeFilament.getNIntervals();
	
	// KJ_DBG(relToRange, activeFilament.getNIntervals());
	
	// Remember that we need one element up and down for the 2nd order interpolation
	if(relToRange < 1 || relToRange >= activeFilament.getNIntervals() - 2) {
		// Advancement would put us out of range. We need to re-map position
		Vec3d tmp = unmap(phi);
		map(tmp, fwd);
		newPhi = unwrap(newPhi);
	}
	
	phi = newPhi;
	return unmap(newPhi);
}

EIGEN_DEVICE_FUNC double FLM::unwrap(double phiIn) {
	double dPhi = phiIn - phi;
	dPhi = fmod(dPhi, 2 * pi);
	dPhi += 3 * pi;
	dPhi = fmod(dPhi, 2 * pi);
	dPhi -= pi;
	
	return phi + dPhi;
}

}