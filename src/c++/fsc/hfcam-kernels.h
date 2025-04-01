#pragma once

#include "kernels/kernels.h"

#include "tensor.h"
#include "kernels/tensor.h"
#include "kernels/message.h"

#include "intersection.h"
#include "geometry.h"

#include <fsc/hfcam.capnp.cu.h>
#include <fsc/index.capnp.cu.h>

namespace fsc {

namespace internal {

struct DensityKernelComputation {
	KDTreeIndexBase tree;
	Eigen::TensorMap<Eigen::Tensor<double, 1>> weights;
	
	cu::DensityKernel::Reader kernel;
	double radius;
	uint32_t kernelDim;
	
	Eigen::TensorMap<Eigen::Tensor<double, 2>> evalPoints;
	unsigned int threadIdx;
	
	double tol;
	
	// Everything below is computed in precompute()
	
	double inverseRadiusSq;
	double normalizedTol;
	double normalizationConstant;
	
	uint32_t nDims;
	
	EIGEN_DEVICE_FUNC inline DensityKernelComputation(
		cu::KDTree::Reader rawTree,
		Eigen::TensorMap<Eigen::Tensor<double, 1>> weights,
		
		cu::DensityKernel::Reader kernel,
		double diameter,
		uint32_t kernelDim,
		
		Eigen::TensorMap<Eigen::Tensor<double, 2>> evalPoints,
		unsigned int threadIdx,
		
		double tol
	) :
		tree(rawTree), weights(weights),
		kernel(kernel), radius(diameter / 2), kernelDim(kernelDim),
		evalPoints(evalPoints), threadIdx(threadIdx),
		tol(tol)
	{
		precompute();
	}

	/**
	 * Computes the unnormalized kernel density function
	 */
	EIGEN_DEVICE_FUNC inline double computeDensityKernelUnnormalized(double distOverRadiusSq) {
		if(kernel.hasGaussian())
			return exp(-0.5*distOverRadiusSq);
		else if(kernel.hasBall())
			return distOverRadiusSq <= 1 ? 1 : 0;
		
		KJ_UNREACHABLE;
	}

	/**
	 * Computes the normalization factor for a kernel density function
	 */
	EIGEN_DEVICE_FUNC inline double computeNormalizationFactor() {
		if(kernel.hasGaussian()) {
			return 1.0 / pow(sqrt(2 * fsc::pi) * radius, kernelDim);
		} else if(kernel.hasBall()) {
			if(kernelDim == 0) return 1.0;
			if(kernelDim == 1) return 1.0 / (2 * radius);
			
			double prev = 1;
			double cur = 2 * radius;
			
			for(uint32_t i = 2; i <= kernelDim; ++i) {
				double next = 2 * fsc::pi / i * radius * radius * prev;
				
				prev = cur;
				cur = next;
			}
			
			return 1 / cur;
		}
		
		KJ_UNREACHABLE;
	}
	
	EIGEN_DEVICE_FUNC inline void precompute() {
		inverseRadiusSq = (1 / radius) * (1 / radius);
		normalizationConstant = computeNormalizationFactor();
		
		// In normalized computation, tolerance should be normalized against the sum of all leaf
		// weights (which is the root weight) and the kernel normalization constant.
		normalizedTol = tol / normalizationConstant / weights(0);
		
		nDims = tree.getNode(0).bounds.size() / 2;
	}
	
	EIGEN_DEVICE_FUNC inline double position(uint64_t dim) {
		return evalPoints(dim, threadIdx);
	}
	
	EIGEN_DEVICE_FUNC inline double handleNode(uint64_t nodeId) {
		// Compute closest and farthest distance
		double closeDistSq = 0;
		double farDistSq = 0;
		
		KDTreeIndexBase::NodeInfo nodeInfo = tree.getNode(nodeId);
		
		for(uint32_t iDim = 0; iDim < nDims; ++iDim) {
			double x = position(iDim);
			
			double low = nodeInfo.bounds[2 * iDim];
			double high = nodeInfo.bounds[2 * iDim + 1];
			
			double shortest = 0;
			double farthest = 0;
			
			if(x < low) {
				shortest = low - x;
				farthest = high - x;
			} else if(x > high) {
				shortest = x - high;
				farthest = x - low;
			} else {
				shortest = std::min(x - low, high - x);
				farthest = std::max(x - low, high - x);
			}
			
			closeDistSq += shortest * shortest;
			farDistSq += farthest * farthest;
		}
		
		// Compute kernel values
		double kernelClose = computeDensityKernelUnnormalized(closeDistSq * inverseRadiusSq);
		double kernelFar = computeDensityKernelUnnormalized(farDistSq * inverseRadiusSq);
		
		// If we are within tolerance, return estimate
		// (Technically, this is a check if weight * discrepancy <= weight * tol, but the weight cancels out)
		if(nodeInfo.node.hasLeaf() || abs(kernelClose - kernelFar) <= tol) {
			return weights(nodeId) * 0.5 * (kernelClose + kernelFar);
		}
		
		// Otherwise, just go through the child nodes
		double result = 0;
		auto interior = nodeInfo.node.getInterior();
		for(uint32_t child = interior.getStart(); child < interior.getEnd(); ++child)
			result += handleNode(child);
		
		return result;
	}
};

}

FSC_DECLARE_KERNEL(
	estimateDensityKernel,
	
	unsigned int,
	
	cu::KDTree::Reader,
	Eigen::TensorMap<Eigen::Tensor<double, 1>>,
	
	Eigen::TensorMap<Eigen::Tensor<double, 2>>,
	cu::DensityKernel::Reader,
	double, uint32_t,
	
	double,
	
	ArrayPtr<double>
);

EIGEN_DEVICE_FUNC void estimateDensityKernel(
	unsigned int idx,
	
	cu::KDTree::Reader tree,
	Eigen::TensorMap<Eigen::Tensor<double, 1>> weights,
	
	Eigen::TensorMap<Eigen::Tensor<double, 2>> evalPoints,
	cu::DensityKernel::Reader kernel,
	double kernelDiam, uint32_t kernelDim,
	
	double tol,
	
	ArrayPtr<double> out
) {
	internal::DensityKernelComputation impl(
		tree, weights,
		kernel, kernelDiam, kernelDim,
		evalPoints, idx,
		tol
	);
	
	out[idx] = impl.handleNode(0) * impl.normalizationConstant;		
}

}
