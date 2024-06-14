#pragma once

#include "kernels/kernels.h"

#include "tensor.h"
#include "kernels/tensor.h"
#include "kernels/message.h"

#include "intersection.h"
#include "geometry.h"


namespace fsc {

FSC_DECLARE_KERNEL(
	rayCastKernel,
	
	Eigen::TensorMap<Eigen::Tensor<double, 2>>,
	Eigen::TensorMap<Eigen::Tensor<double, 2>>,
	cu::MergedGeometry::Reader,
	cu::IndexedGeometry::Reader,
	cu::IndexedGeometry::IndexData::Reader,
	
	ArrayPtr<IntersectResult>
);

EIGEN_DEVICE_FUNC void rayCastKernel(
	unsigned int idx,
	
	Eigen::TensorMap<Eigen::Tensor<double, 2>> pStart,
	Eigen::TensorMap<Eigen::Tensor<double, 2>> pEnd,
	cu::MergedGeometry::Reader geo,
	cu::IndexedGeometry::Reader index,
	cu::IndexedGeometry::IndexData::Reader indexData,
	
	ArrayPtr<IntersectResult> out
) {
	Vec3d p1(pStart(idx, 0), pStart(idx, 1), pStart(idx, 2));
	Vec3d p2(pEnd(idx, 0), pEnd(idx, 1), pEnd(idx, 2));
	
	out[idx] = intersectGeometryFirstHit(p1, p2, geo, index, indexData);
}

}