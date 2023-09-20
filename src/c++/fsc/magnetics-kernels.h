#pragma once

#include <fsc/magnetics.capnp.cu.h>

#include "grids.h"

#include "kernels/kernels.h"

namespace fsc { namespace kernels {

using Field = Eigen::Tensor<double, 4>;
using FieldRef = Eigen::TensorMap<Field>;

using MFilament = Eigen::Tensor<double, 2>;
using FilamentRef = Eigen::TensorMap<MFilament>;

FSC_DECLARE_KERNEL(addFieldKernel, FieldRef, FieldRef, double);
FSC_DECLARE_KERNEL(addFieldInterpKernel, FieldRef, ToroidalGridStruct, FieldRef, ToroidalGridStruct, double);
FSC_DECLARE_KERNEL(biotSavartKernel, ToroidalGridStruct, FilamentRef, double, double, double, FieldRef);
FSC_DECLARE_KERNEL(eqFieldKernel, ToroidalGridStruct, cu::AxisymmetricEquilibrium::Reader, double, FieldRef);

}}