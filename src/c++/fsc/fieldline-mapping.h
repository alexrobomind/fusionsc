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

struct RFLM {
	//! Computes the real-space coordinates of the last-mapped position projected to the given phi plane
	inline EIGEN_DEVICE_FUNC Vec3d unmap(double phi);
	
	//! Captures a position for mapping using the closest mapping filament
	inline EIGEN_DEVICE_FUNC void map(const Vec3d& x, bool goingCcw);
	
	//! Advances to a new phi position, remapping if neccessary. Returns new real-space position
	// Note: No guarantees that flm.phi == newPhi
	inline EIGEN_DEVICE_FUNC Vec3d advance(double newPhi);
	
	inline EIGEN_DEVICE_FUNC double getFieldlinePosition(double phi);
	inline EIGEN_DEVICE_FUNC void setFieldlinePosition(double newValue);
	
	inline EIGEN_DEVICE_FUNC RFLM(cu::ReversibleFieldlineMapping mapping);
	
	cu::ReversibleFieldlineMapping mapping;
	Vec2d uv;
	uint64_t currentSection;
	double phi;
	double lenOffset;
	
	uint32_t nPad;
	
	// Cached information from mapping about current section shape
	uint64_t nPhi;
	uint64_t nZ;
	uint64_t nR;
	double phi1;
	double phi2;
	
	struct TensorField {
		cu::Float64Tensor reader;
		RFLM& parent;
		
		TensorField(cu::Float64Tensor reader, RFLM& parent) :
			reader(reader), parent(parent)
		{}
		
		double operator()(int iPhi, int iZ, int iR) const {
			iPhi += parent.nPad;
			
			if(iPhi < 0) iPhi = 0;
			if(iPhi >= parent.nPhi) iPhi = parent.nPhi - 1;
			if(iZ < 0) iZ = 0;
			if(iZ >= parent.nZ) iZ = parent.nZ - 1;
			if(iR < 0) iR = 0;
			if(iR >= parent.nR) iR = parent.nR - 1;
			
			int iLinear = iR + parent.nR * (iZ + parent.nZ * iPhi);
			return reader.getData()[iLinear];
		}
	};

private:
	inline void activateSection(uint64_t iSection);
	inline cu::ReversibleFieldlineMapping::Section activeSection();
	
	inline EIGEN_DEVICE_FUNC void interpolate(double phi, Vec2d& rz, Mat2d& jacobian);
	
	inline static double unwrap(double phiWrapped);
};

// === class RFLM ===

EIGEN_DEVICE_FUNC RFLM::RFLM(cu::ReversibleFieldlineMapping mapping) :
	mapping(mapping),
	uv(0.5, 0.5),
	currentSection(0),
	phi(0), lenOffset(0),
	
	nPad(mapping.getNPad()),
	
	nPhi(0), nZ(0), nR(0)
{}
	
double RFLM::unwrap(double dphi) {
	dphi = fmod(dphi, 2 * pi);
	dphi += 2 * pi;
	dphi = fmod(dphi, 2 * pi);
	return dphi;
}

EIGEN_DEVICE_FUNC void RFLM::activateSection(uint64_t iSection) {
	currentSection = iSection;
	
	auto section = mapping.getSections()[currentSection];
	auto shape = section.getR().getShape();
	nPhi = shape[0];
	nZ = shape[1];
	nR = shape[2];
	
	if(mapping.getSections().size() == 1) {
		phi1 = mapping.getSurfaces()[0];
		phi2 = phi1 + 2 * pi;
	} else {	
		phi1 = mapping.getSurfaces()[iSection];
		phi2 = mapping.getSurfaces()[(iSection + 1) % mapping.getSurfaces().size()];
		
		phi2 = phi1 + unwrap(phi2 - phi1);
	}
	
	// KJ_DBG("Activated section", iSection, phi1, phi2, nPhi, nZ, nR);
}

EIGEN_DEVICE_FUNC cu::ReversibleFieldlineMapping::Section RFLM::activeSection() {
	return mapping.getSections()[currentSection];
}

EIGEN_DEVICE_FUNC double RFLM::getFieldlinePosition(double phi) {
	using Strategy = C1CubicInterpolation<double>;
	using Interpolator = NDInterpolator<3, Strategy>;
	TensorField lenField(activeSection().getTraceLen(), *this);
	
	Interpolator interpolator(
		Strategy(),
		{ Interpolator::Axis(0, phi2 - phi1, nPhi - 2 * nPad - 1), Interpolator::Axis(0, 1, nZ - 1), Interpolator::Axis(0, 1, nR - 1) }
	);
	
	Vec3d interpCoords(unwrap(phi - phi1), uv(0), uv(1));
	return interpolator(lenField, interpCoords) + lenOffset;
}

EIGEN_DEVICE_FUNC void RFLM::setFieldlinePosition(double newVal) {
	lenOffset += newVal - getFieldlinePosition(this -> phi);
}

EIGEN_DEVICE_FUNC Vec3d RFLM::unmap(double phi) {
	using Strategy = C1CubicInterpolation<double>;
	using Interpolator = NDInterpolator<3, Strategy>;
	
	TensorField rField(activeSection().getR(), *this);
	TensorField zField(activeSection().getZ(), *this);
	
	Interpolator interpolator(
		Strategy(),
		{ Interpolator::Axis(0, phi2 - phi1, nPhi - 2 * nPad - 1), Interpolator::Axis(0, 1, nZ - 1), Interpolator::Axis(0, 1, nR - 1) }
	);
	
	Vec3d interpCoords(unwrap(phi - phi1), uv(0), uv(1));
	double rVal = interpolator(rField, interpCoords);
	double zVal = interpolator(zField, interpCoords);
	
	// KJ_DBG("Unmapping", phi, uv(0), uv(1), rVal, zVal);
	
	return Vec3d(rVal * cos(phi), rVal * sin(phi), zVal);
}

EIGEN_DEVICE_FUNC void RFLM::map(const Vec3d& x, bool ccw) {
	// --- Select active section for inversion ---
	phi = atan2(x[1], x[0]);
	
	auto phiPlanes = mapping.getSurfaces();
	
	// KJ_DBG("Selecting section");
	size_t iSection;
	for(iSection = 0; iSection < phiPlanes.size(); ++iSection) {
		// We shift our sections slightly against the direction we are
		// travelling towards, and shrinking it in the other. This guarantees that
		// right on the mapping planes, we always map into the adequate section
		// for the direction we are going towards.
		double dirShift = ccw ? 0.001 : -0.001;
		double phiStart = phiPlanes[iSection] + dirShift;
		double phiEnd = phiPlanes[(iSection + 1) % phiPlanes.size()] + dirShift;
		
		double d1 = unwrap(phi - phiStart);
		double d2 = unwrap(phiEnd - phiStart);
		
		// If we have only a single section, that one spans the entire range.
		// In this case, d2 is 0 and we would not select any section.
		
		if(d1 < d2 || phiPlanes.size() == 1)
			break;
	}
		
	// KJ_DBG(iSection);
	activateSection(iSection);
	
	// --- Perform Newton-style inversion in active section ---
	using ADS = Eigen::AutoDiffScalar<Vec2d>;
	using Strategy = C1CubicInterpolation<ADS>;
	using Interpolator = NDInterpolator<3, Strategy>;
	
	TensorField rField(activeSection().getR(), *this);
	TensorField zField(activeSection().getZ(), *this);
	
	Interpolator interpolator(
		Strategy(),
		{ Interpolator::Axis(0, phi2 - phi1, nPhi - 2 * nPad - 1), Interpolator::Axis(0, 1, nZ - 1), Interpolator::Axis(0, 1, nR - 1) }
	);
	
	double rRef = sqrt(x[0] * x[0] + x[1] * x[1]);
	double zRef = x[2];
	
	for(size_t i = 0; i < 10; ++i) {
		// Calculate values and derivatives for r and z
		Vec3<ADS> interpCoords(unwrap(phi - phi1), ADS(uv(0), 2, 0), ADS(uv(1), 2, 1));
		ADS rVal = interpolator(rField, interpCoords);
		ADS zVal = interpolator(zField, interpCoords);
		
		Vec2d dx(rRef - rVal.value(), zRef - zVal.value());
		
		double drdu = rVal.derivatives()(0);
		double dzdu = zVal.derivatives()(0);
		double drdv = rVal.derivatives()(1);
		double dzdv = zVal.derivatives()(1);
		
		double invDet = 1 / (drdu * dzdv - dzdu * drdv);
		double du = invDet * (dx(0) * dzdv - dx(1) * drdv);
		double dv = invDet * (drdu * dx(1) - dzdu * dx(0));
		
		uv(0) += du;
		uv(1) += dv;
	}
	
	// KJ_DBG("Map completed", uv(0), uv(1), phi);
}

EIGEN_DEVICE_FUNC Vec3d RFLM::advance(double newPhi) {
	bool fwd = newPhi > phi;
	
	while(true) {		
		double phiClamped = newPhi;
		bool remap = false;
		
		constexpr double shiftTol = 1e-5;
		
		// Check if the new point is in the bounds of the current section
		// Assumes the current point is in the section
		if(fwd) {
			double dToEnd = unwrap(phi2 - phi);
			double dToTarget = newPhi - phi;
			
			if(dToEnd < dToTarget) {
				phiClamped = phi + dToEnd + shiftTol;
				remap = true;
			}
		} else {
			double dToStart = unwrap(phi - phi1);
			double dToTarget = phi - newPhi;
			
			if(dToStart < dToTarget) {
				phiClamped = phi - dToStart - shiftTol;
				remap = true;
			}
		}
		
		if(remap) {
			// KJ_DBG("Remapping", phi, newPhi, phiClamped);
			Vec3d tmp = unmap(phiClamped);
			double len = getFieldlinePosition(phiClamped);
			map(tmp, fwd);
			phi = phiClamped;
			setFieldlinePosition(len);
		} else {
			phi = newPhi;
			// KJ_DBG(phi);
			return unmap(newPhi);
		}
	}
}

// === class FLM ===

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