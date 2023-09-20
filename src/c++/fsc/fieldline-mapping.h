#pragma once

#include <fsc/flt.capnp.cu.h>

#include "interpolation.h"
#include "index.h"

#include <iostream>

namespace fsc {
	
Mapper::Client newMapper(FLT::Client flt, KDTreeService::Client indexer);

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
	
	inline EIGEN_DEVICE_FUNC RFLM(cu::ReversibleFieldlineMapping::Reader mapping);
	inline EIGEN_DEVICE_FUNC RFLM(const RFLM& other) = default;
	
	cu::ReversibleFieldlineMapping::Reader mapping;
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
		cu::Float64Tensor::Reader reader;
		RFLM& parent;
		
		TensorField(cu::Float64Tensor::Reader reader, RFLM& parent) :
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
			
			if(iLinear >= reader.getData().size()) {
				KJ_DBG("Bad iLinear", iLinear, reader.getData().size(), iZ, iR, iPhi);
			}
			
			return reader.getData()[iLinear];
		}
	};

private:
	inline void activateSection(uint64_t iSection);
	inline cu::ReversibleFieldlineMapping::Section::Reader activeSection();
	
	inline EIGEN_DEVICE_FUNC void interpolate(double phi, Vec2d& rz, Mat2d& jacobian);
	
	inline static double unwrap(double phiWrapped);
	
	static constexpr double SECTION_TOL = 0.001;
};

// === class RFLM ===

EIGEN_DEVICE_FUNC RFLM::RFLM(cu::ReversibleFieldlineMapping::Reader mapping) :
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

EIGEN_DEVICE_FUNC cu::ReversibleFieldlineMapping::Section::Reader RFLM::activeSection() {
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
	
	double phiCoord = unwrap(phi - phi1 + 2 * SECTION_TOL) - 2 * SECTION_TOL;
	
	Vec3d interpCoords(phiCoord, uv(0), uv(1));
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
	
	double phiCoord = unwrap(phi - phi1 + 2 * SECTION_TOL) - 2 * SECTION_TOL;
	
	Vec3d interpCoords(phiCoord, uv(0), uv(1));
	double rVal = interpolator(rField, interpCoords);
	double zVal = interpolator(zField, interpCoords);
	
	// KJ_DBG("Unmapping", phi, uv(0), uv(1), rVal, zVal);
	
	return Vec3d(rVal * cos(phi), rVal * sin(phi), zVal);
}

EIGEN_DEVICE_FUNC void RFLM::map(const Vec3d& x, bool ccw) {
	// --- Select active section for inversion ---
	phi = atan2(x[1], x[0]);
	
	// Safe handling for "NaN" case
	if(phi != phi) {
		activateSection(0);
		uv(0) = phi;
		uv(1) = phi;
		return;
	}
	
	auto phiPlanes = mapping.getSurfaces();
	
	// KJ_DBG("Selecting section");
	size_t iSection;
	for(iSection = 0; iSection < phiPlanes.size(); ++iSection) {
		// We shift our sections slightly against the direction we are
		// travelling towards, and shrinking it in the other. This guarantees that
		// right on the mapping planes, we always map into the adequate section
		// for the direction we are going towards.
		double dirShift = ccw ? -SECTION_TOL : SECTION_TOL;
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
	
	
	uv(0) = activeSection().getU0();
	uv(1) = activeSection().getV0();
	
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
	
	double phiCoord = unwrap(phi - phi1 + 2 * SECTION_TOL) - 2 * SECTION_TOL;
	
	for(size_t i = 0; i < 50; ++i) {
		// Calculate values and derivatives for r and z
		Vec3<ADS> interpCoords(phiCoord, ADS(uv(0), 2, 0), ADS(uv(1), 2, 1));
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
		
		//double scale = std::min(((double) i + 1) / 20, 1.0);
		double scale = 0.05 / sqrt(du * du + dv * dv);
		if(scale > 1)
			scale = 1;
		
		
		uv(0) += scale * du;
		uv(1) += scale * dv;
		
		if(dx.norm() < 1e-12)
			break;
		// KJ_DBG(du, dv, dx(0), dx(1), rVal.value(), zVal.value());
	}
	
	// KJ_DBG("Map completed", rRef, zRef, uv(0), uv(1), phi);
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
				// KJ_DBG(phi, newPhi, dToEnd, dToTarget);
				phiClamped = phi + dToEnd + shiftTol;
				remap = true;
			}
		} else {
			double dToStart = unwrap(phi - phi1);
			double dToTarget = phi - newPhi;
			
			if(dToStart < dToTarget) {
				// KJ_DBG(phi, newPhi, dToStart, dToTarget);
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

}