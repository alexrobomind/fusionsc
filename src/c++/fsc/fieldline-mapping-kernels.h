#pragma once

#include <fsc/flt.capnp.cu.h>

#include "kernels/kernels.h"

#include "tensor.h"
#include "kernels/tensor.h"
#include "kernels/message.h"

#include "interpolation.h"
#include "fieldline-mapping.h"

namespace fsc {
	
using Tensor3Ref = Eigen::TensorMap<Eigen::Tensor<double, 3>>;
using Tensor2Ref = Eigen::TensorMap<Eigen::Tensor<double, 2>>;

FSC_DECLARE_KERNEL(
	invertRflmKernel,
	
	Tensor3Ref, Tensor3Ref,
	Tensor3Ref, Tensor3Ref,
	
	double, double, double, double
);


/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void invertRflmKernel(unsigned int idx, Tensor3Ref rVals, Tensor3Ref zVals, Tensor3Ref uVals, Tensor3Ref vVals, double rMin, double rMax, double zMin, double zMax) {
	const unsigned int nU = rVals.dimension(0);
	const unsigned int nV = rVals.dimension(1);
	
	const unsigned int nR = uVals.dimension(0);
	const unsigned int nZ = uVals.dimension(1);
	const unsigned int nPhi = uVals.dimension(2);
	
	const unsigned int iPhi = idx / (nZ * nR);
	const unsigned int iZ = (idx - iPhi * nZ * nR) / nR;
	const unsigned int iR = (idx - iPhi * nZ * nR - iZ * nR);
	
	// Compute position
	const double r = rMin + (rMax - rMin) * iR / (nR - 1);
	const double z = zMin + (zMax - zMin) * iZ / (nZ - 1);
			
	// Prepare interpolator
	using ADS = Eigen::AutoDiffScalar<Vec2d>;
	using Strategy = C1CubicInterpolation<ADS>;
	using Interpolator = NDInterpolator<2, Strategy>;
	
	Interpolator interpolator(
		Strategy(),
		{ Interpolator::Axis(0, 1, nU - 1), Interpolator::Axis(0, 1, nV - 1) }
	);
	
	// Run interpolation loop
	auto clamp = [&](int& iU, int& iV) {
		if(iU < 0) iU = 0;
		if(iU >= nU) iU = nU - 1;
		if(iV < 0) iV = 0;
		if(iV >= nV) iV = nV - 1;
	};
	
	auto interpZ = [&](int iU, int iV) {
		clamp(iU, iV);
		return zVals(iU, iV, iPhi);
	};
	auto interpR = [&](int iU, int iV) {
		clamp(iU, iV);
		return rVals(iU, iV, iPhi);
	};
	
	// Copied from 
	double u = 0.5;
	double v = 0.5;
	
	for(size_t i = 0; i < 50; ++i) {
		// Calculate values and derivatives for r and z
		Vec2<ADS> interpCoords(ADS(u, 2, 0), ADS(v, 2, 1));
		ADS rVal = interpolator(interpR, interpCoords);
		ADS zVal = interpolator(interpZ, interpCoords);
		
		Vec2d dx(r - rVal.value(), z - zVal.value());
		
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
		
		u += scale * du;
		v += scale * dv;
		
		if(dx.norm() < 1e-12)
			break;
		// KJ_DBG(du, dv, dx(0), dx(1), rVal.value(), zVal.value());
	}
	
	uVals(iR, iZ, iPhi) = u;
	vVals(iR, iZ, iPhi) = v;
}

EIGEN_DEVICE_FUNC inline void mapKernel(unsigned int idx, Tensor2Ref in, cu::RFLMKernelData::Builder out, cu::ReversibleFieldlineMapping::Reader mapping) {
	double x = in(idx, 0);
	double y = in(idx, 1);
	double z = in(idx, 2);
	
	RFLM m(mapping);
	m.map(Vec3d(x, y, z), true);
	m.save(out.mutateStates()[idx]);
}

EIGEN_DEVICE_FUNC inline void unmapKernel(unsigned int idx, cu::RFLMKernelData::Reader in, Tensor2Ref out, cu::ReversibleFieldlineMapping::Reader mapping) {	
	RFLM m(mapping);
	m.load(in.getStates()[idx]);
	
	auto pv = in.getPhiValues();
	
	double phi;
	if(pv.size() > 0)
		phi = pv[idx];
	else
		phi = m.phi;
	
	Vec3d pos = m.unmap(phi);
	out(idx, 0) = pos(0);
	out(idx, 1) = pos(1);
	out(idx, 2) = pos(2);
}

}