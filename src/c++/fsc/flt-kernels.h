#pragma once

#include <fsc/flt.capnp.cu.h>

#include "kernels/kernels.h"

namespace fsc {

FSC_DECLARE_KERNEL(
	fltKernel,
	
	fsc::cu::FLTKernelData::Builder,
	Eigen::TensorMap<Eigen::Tensor<double, 4>>,
	fsc::cu::FLTKernelRequest::Builder,
	
	fsc::cu::MergedGeometry::Reader,
	fsc::cu::IndexedGeometry::Reader,
	fsc::cu::IndexedGeometry::IndexData::Reader,
	
	fsc::cu::ReversibleFieldlineMapping::Reader
);

}