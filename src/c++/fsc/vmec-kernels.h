#pragma once

#include "kernels/kernels.h"

#include <fsc/vmec.capnp.cu.h>

namespace fsc {

FSC_DECLARE_KERNEL(
	computeSurfaceKernel,
	
	cu::VmecKernelComm::Builder
);

namespace internal { namespace {

EIGEN_DEVICE_FUNC inline void computeRZ(
	cu::VmecSurfaces::Reader surfaces,
	
	double s, double phi, double theta,
	double& rOut, double& zOut
) {
	int32_t nTor = (int32_t) surfaces.getNTor();
	uint32_t mPol = surfaces.getMPol();
	uint32_t nfp = surfaces.getPeriod();
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
	
	KJ_DBG(n1, n2, w);
		
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

}	