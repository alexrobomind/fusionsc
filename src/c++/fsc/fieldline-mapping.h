#pragma once

#include <fsc/flt.capnp.cu.h>

#include "interpolation.h"
#include "index.h"

#include <iostream>

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
	// using Strategy = LinearInterpolation<double>;
	using Interpolator = NDInterpolator<1, Strategy>;
	
	auto data = activeFilament.getData();
	
	
	double phi1 = activeFilament.getPhiStart();
	double phi2 = activeFilament.getPhiEnd();
	double scaled = (phi - phi1) / (phi2 - phi1) * activeFilament.getNIntervals();
	
	int base = floor(scaled);
	double rem = scaled - base;
	
	std::array<double, 6> outData;
	
	Strategy strat;
	auto coeffs = strat.coefficients(rem);
	for(size_t j = 0; j < 6; ++j) {
		double accum = 0;
		for(size_t i = 0; i < 4; ++i) {
			accum += coeffs[i] * data[6 * (i + base - 1) + j];
		}
		outData[j] = accum;
	}
	
	rz[0] = outData[0];
	rz[1] = outData[1];
	
	for(size_t i = 0; i < 4; ++i)
		jacobian.data()[i] = outData[i + 2];
	
	/*
	Interpolator::Axis ax1(activeFilament.getPhiStart(), activeFilament.getPhiEnd(), activeFilament.getNIntervals());
	Interpolator interp(Strategy(), {ax1});
	
	for(int dim = 0; dim < 2; ++dim) {
		auto filamentPos = [&](int i) {
			KJ_ASSERT(i <= activeFilament.getNIntervals());
			KJ_ASSERT(i >= 0);
			// KJ_DBG(i, 6 * i + dim, data.size());
			return data[6 * i + dim];
		};
		rz(dim) = interp(filamentPos, Vec1d {phi});
	}
	
	for(int idx = 0; idx < 4; ++idx) {
		auto filamentJacobian = [&](int i) {
			// KJ_DBG(i, 6 * i + idx + 2, data.size());
			return data[6 * i + idx + 2];
		};
		jacobian.data()[idx] = interp(filamentJacobian, Vec1d {phi});
	}
	*/
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
	
	// KJ_DBG(filamentIdx, pointIdx, findResult.distance);
	
	activeFilament = dir.getFilaments()[filamentIdx];
	
	// Compute phi baseline
	double phiBase = activeFilament.getPhiStart() + ((activeFilament.getPhiEnd() - activeFilament.getPhiStart()) / activeFilament.getNIntervals()) * pointIdx;
	
	// Unwrap phi using phiBase
	phi = phiBase;
	phi = unwrap(newPhi);
	// KJ_DBG(phiBase, phi, activeFilament.getPhiStart(), activeFilament.getPhiEnd(), activeFilament.getNIntervals());
	
	// Interpolate mapping to active plane
	Vec2d filRZ;
	Mat2d filJacobian;
	interpolate(phi, filRZ, filJacobian);
	
	// Compute mapped position
	double r = sqrt(x[0] * x[0] + x[1] * x[1]);
	double z = x[2];
	
	// std::cout << filRZ << std::endl << filJacobian << std::endl;
	
	// KJ_DBG(r, z, filRZ[0], filRZ[1]);
	
	uv = filJacobian.inverse() * (Vec2d { r, z } - filRZ);
	
	// KJ_DBG(uv[0], uv[1]);
}

EIGEN_DEVICE_FUNC Vec3d FLM::advance(double newPhi, bool fwd) {
	double relToRange = (newPhi - activeFilament.getPhiStart()) / (activeFilament.getPhiEnd() - activeFilament.getPhiStart()) * activeFilament.getNIntervals();
	
	// KJ_DBG(relToRange, activeFilament.getNIntervals());
	
	// Remember that we need one element up and down for the 2nd order interpolation
	if(relToRange < 1 || relToRange >= activeFilament.getNIntervals() - 2) {
		// KJ_DBG("Remapping", relToRange, activeFilament.getNIntervals(), activeFilament.getPhiStart(), activeFilament.getPhiEnd());
		// Advancement would put us out of range. We need to re-map position
		Vec3d tmp = unmap(phi);
		double phiPrev = phi;
		map(tmp, fwd);
		
		newPhi = unwrap(newPhi);
		
		// ("Remapped", phiPrev, phi, newPhi);
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