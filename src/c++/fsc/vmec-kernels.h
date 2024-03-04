#pragma once

#include "kernels/kernels.h"

#include <fsc/vmec.capnp.cu.h>

#include <iostream>

namespace fsc {

FSC_DECLARE_KERNEL(
	computeSurfaceKernel,
	
	cu::VmecKernelComm::Builder
);

FSC_DECLARE_KERNEL(
	invertSurfaceKernel,
	
	cu::VmecKernelComm::Builder
);

namespace internal { namespace {

EIGEN_DEVICE_FUNC inline void computeRZ(
	cu::FourierSurfaces::Reader surfaces,
	
	double s, double phi, double theta,
	double& rOut, double& zOut
) {
	int32_t nTor = (int32_t) surfaces.getNTor();
	uint32_t mPol = surfaces.getMPol();
	uint32_t nfp = surfaces.getToroidalSymmetry();
	uint32_t nSurf = surfaces.getRCos().getShape()[0];
	
	uint32_t numTorCoeffs = 2 * nTor + 1;
	uint32_t numPolCoeffs = mPol + 1;
	
	// Calculates the linear index of a mode in the mode array
	auto linearIndex = [&](uint32_t iSurf, int32_t n, uint32_t m) {
		uint32_t result = iSurf;
		
		result *= numTorCoeffs;
		result += n >= 0 ? n : numTorCoeffs + n;
		
		result *= numPolCoeffs;
		result += m;
		
		return result;
	};
	
	double r = 0;
	double z = 0;
	
	// Calculate interpolation coordinates
	uint32_t n1;
	uint32_t n2;
	double w;
	
	if(s < 1) {
		double dr = 1.0 / (nSurf - 1);
		size_t lower = (size_t)(s / dr);
		double remainder = fmod(s / dr, 1.0);
		
		n1 = lower;
		n2 = lower + 1;
		w = remainder;
	} else {
		n1 = 0;
		n2 = nSurf - 1;
		w = s;
	}
		
	auto rCos = surfaces.getRCos().getData();
	auto zSin = surfaces.getZSin().getData();
	
	for(int32_t n = -nTor; n <= nTor; ++n) {		
		for(uint32_t m = 0; m <= mPol; ++m) {
			auto i1 = linearIndex(n1, n, m);
			auto i2 = linearIndex(n2, n, m);
			
			double angle = m * theta - n * phi;
			
			r += ((1 - w) * rCos[i1] + w * rCos[i2]) * cos(angle);
			z += ((1 - w) * zSin[i1] + w * zSin[i2]) * sin(angle);
		}
	}
	
	if(surfaces.isNonSymmetric()) {
		auto nonsym = surfaces.getNonSymmetric();
		
		auto zCos = nonsym.getZCos().getData();
		auto rSin = nonsym.getRSin().getData();
		
		for(int32_t n = -nTor; n <= nTor; ++n) {			
			for(uint32_t m = 0; m <= mPol; ++m) {
				auto i1 = linearIndex(n1, n, m);
				auto i2 = linearIndex(n2, n, m);
				
				double angle = m * theta - n * phi;
				
				r += ((1 - w) * rSin[i1] + w * rSin[i2]) * sin(angle);
				z += ((1 - w) * zCos[i1] + w * zCos[i2]) * cos(angle);
			}
		}
	}
	
	rOut = r;
	zOut = z;
}

EIGEN_DEVICE_FUNC inline void computeRZFromVxVy(
	cu::FourierSurfaces::Reader surfaces,
	
	double phi, double vx, double vy,
	double& rOut, double& zOut
) {
	double s = sqrt(vx * vx + vy * vy);
	double theta = atan2(vy, vx);
	
	if(s < 1e-5)
		theta = 0;
	
	computeRZ(surfaces, s, phi, theta, rOut, zOut);
}

}}

EIGEN_DEVICE_FUNC inline void computeSurfaceKernel(
	unsigned int idx,
	
	cu::VmecKernelComm::Builder comm
) {
	auto posIn = comm.getSpt();
	size_t blockSize = posIn.size() / 3;
	
	double s = posIn[idx + 0 * blockSize];
	double phi = posIn[idx + 1 * blockSize];
	double theta = posIn[idx + 2 * blockSize];
	
	double r; double z;
	internal::computeRZ(comm.getSurfaces(), s, phi, theta, r, z);
	
	auto posOut = comm.mutatePzr();
	posOut.set(idx + 0 * blockSize, phi);
	posOut.set(idx + 1 * blockSize, z);
	posOut.set(idx + 2 * blockSize, r);
}

EIGEN_DEVICE_FUNC inline void invertSurfaceKernel(
	unsigned int idx,
	
	cu::VmecKernelComm::Builder comm
) {
	auto posIn = comm.getPzr();
	size_t blockSize = posIn.size() / 3;
	
	double phi = posIn[idx + 0 * blockSize];
	double z   = posIn[idx + 1 * blockSize];
	double r   = posIn[idx + 2 * blockSize];
	
	double vx = 0;
	double vy = 0;
	
	// Iterate over successively smaller linearization region
	unsigned int i = 0;
	for(double h = 0.3; h > 1e-6 && i < 10; ++i) {
		//std::cout << "C: " << c << std::endl;
		//std::cout << "h: " << h << std::endl;

		// Calculate 4-point star around current point
		double starX[4] = { vx - h, vx + h, vx, vx };
		double starY[4] = { vy, vy, vy - h, vy + h };
		
		double starR[4];
		double starZ[4];
		double rAv = 0; double zAv = 0;
		for(unsigned int i = 0; i < 4; ++i) {
			internal::computeRZFromVxVy(comm.getSurfaces(), phi, starX[i], starY[i], starR[i], starZ[i]);
			rAv += starR[i];
			zAv += starZ[i];
		}
		
		rAv *= 0.25;
		zAv *= 0.25;
		
		// Calculate Jacobian
		Mat2d jacobian;
		jacobian(0, 0) = starR[1] - starR[0];
		jacobian(1, 0) = starZ[1] - starZ[0];
		jacobian(0, 1) = starR[3] - starR[2];
		jacobian(1, 1) = starZ[3] - starZ[2];
		jacobian /= 2 * h;
		
		if(jacobian.determinant() < 1e-8) {
			jacobian.setIdentity();
		}
		
		// Mat2d invJacobian = jacobian.inverse();
		Vec2d deltaRZ(r - rAv, z - zAv);
		Vec2d deltaXY = jacobian.inverse() * deltaRZ;
		
		vx += deltaXY[0];
		vy += deltaXY[1];
		
		h = deltaXY.norm();
	}
	
	double s = sqrt(vx * vx + vy * vy);
	double theta = s > 1e-5 ? atan2(vy, vx) : 0;
	
	auto posOut = comm.mutateSpt();
	posOut.set(idx + 0 * blockSize, s);
	posOut.set(idx + 1 * blockSize, phi);
	posOut.set(idx + 2 * blockSize, theta);
	
}

}	