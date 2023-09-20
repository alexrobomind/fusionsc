#pragma once

#include "../eigen.h"
#include "device.h"

namespace fsc {
	
template<typename TensorType>
struct TensorMapping : public DeviceMapping<kj::Array<typename TensorType::Scalar>> {
	using Scalar = typename TensorType::Scalar;
	
	TensorMap<TensorType> hostMap;
	TensorMap<TensorType> deviceMap;
	
	TensorMapping(Held<TensorType> tensor, DeviceBase& device, bool allowAlias) :
		DeviceMapping<kj::Array<Scalar>>(kj::ArrayPtr<Scalar>(tensor -> data(), tensor -> size()).attach(tensor.x()), device, allowAlias),
		hostMap(*tensor),
		deviceMap(DeviceMapping<kj::Array<typename TensorType::Scalar>>::get().begin(), tensor -> dimensions())
	{}
	
	TensorMapping(Held<TensorMap<TensorType>> tensor, DeviceBase& device, bool allowAlias) :
		DeviceMapping<kj::Array<Scalar>>(kj::ArrayPtr<Scalar>(tensor -> data(), tensor -> size()).attach(tensor.x()), device, allowAlias),
		hostMap(*tensor),
		deviceMap(DeviceMapping<kj::Array<Scalar>>::get(), tensor -> dimensions())
	{}
	
	TensorMap<TensorType> get() { return deviceMap; }
	TensorMap<TensorType> getHost() { return hostMap; }
};

template<typename TensorType>
struct ConstTensorMapping : public DeviceMapping<kj::Array<const typename TensorType::Scalar>> {
	using Scalar = typename TensorType::Scalar;
	
	TensorMap<const TensorType> hostMap;
	TensorMap<TensorType> deviceMap;
	
	ConstTensorMapping(Held<TensorMap<const TensorType>> tensor, DeviceBase& device, bool allowAlias) :
		DeviceMapping<kj::Array<const Scalar>>(kj::ArrayPtr<const Scalar>(tensor -> data(), tensor -> size()).attach(tensor.x()), device, allowAlias),
		hostMap(*tensor),
		deviceMap(DeviceMapping<kj::Array<const Scalar>>::get().begin(), tensor -> dimensions())
	{}
	
	TensorMap<TensorType> get() { return deviceMap; }
	TensorMap<const TensorType> getHost() { return hostMap; }
};

template<typename TVal, int tRank, int tOpts, typename Index>
struct DeviceMapping<Tensor<TVal, tRank, tOpts, Index>>
	: public TensorMapping<Tensor<TVal, tRank, tOpts, Index>>
{
	DeviceMapping(Tensor<TVal, tRank, tOpts, Index> t, DeviceBase& device, bool allowAlias) :
		TensorMapping<Tensor<TVal, tRank, tOpts, Index>>(
			heapHeld<Tensor<TVal, tRank, tOpts, Index>>(mv(t)),
			device,
			allowAlias
		)
	{}
};

template<typename TVal, typename Dims, int options, typename Index>
struct DeviceMapping<TensorFixedSize<TVal, Dims, options, Index>>
	: public TensorMapping<TensorFixedSize<TVal, Dims, options, Index>>
{
	DeviceMapping(TensorFixedSize<TVal, Dims, options, Index> t, DeviceBase& device, bool allowAlias) :
		TensorMapping<TensorFixedSize<TVal, Dims, options, Index>>(
			heapHeld<TensorFixedSize<TVal, Dims, options, Index>>(mv(t)),
			device,
			allowAlias
		)
	{}
};

template<typename T>
struct DeviceMapping<Own<TensorMap<T>>>
	: public TensorMapping<T>
{
	DeviceMapping(Own<TensorMap<T>> t, DeviceBase& device, bool allowAlias) :
		TensorMapping<T>(
			ownHeld<TensorMap<T>>(mv(t)),
			device,
			allowAlias
		)
	{}
};

template<typename T>
struct DeviceMapping<Own<TensorMap<const T>>>
	: public ConstTensorMapping<T>
{
	DeviceMapping(Own<TensorMap<const T>> t, DeviceBase& device, bool allowAlias) :
		ConstTensorMapping<T>(
			ownHeld<TensorMap<const T>>(mv(t)),
			device,
			allowAlias
		)
	{}
};

template<typename TVal, typename Dims, int options, typename Index>
struct DeviceMapping<Own<TensorMap<const TensorFixedSize<TVal, Dims, options, Index>>>>
	: public ConstTensorMapping<TensorFixedSize<TVal, Dims, options, Index>>
{
	DeviceMapping(Own<TensorMap<const TensorFixedSize<TVal, Dims, options, Index>>> t, DeviceBase& device, bool allowAlias) :
		ConstTensorMapping<TensorFixedSize<TVal, Dims, options, Index>>(
			ownHeld<TensorMap<const TensorFixedSize<TVal, Dims, options, Index>>>(mv(t)),
			device,
			allowAlias
		)
	{}
};

}