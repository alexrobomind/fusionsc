#pragma once

# include <thread>
# include <kj/debug.h>
# include <kj/memory.h>
# include <capnp/endian.h>

#include "eigen.h"
#include "kernels.h"
#include "device.h"
#include "memory.h"

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	#define FSC_WIRE_MATCHES_NATIVE
#endif

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

// ===================== Reads and writes form serialized tensors ========================

template<typename T, int rank, int options, typename Index, typename T2>
void readTensor(T2 reader, Tensor<T, rank, options, Index>& out) {
	using TensorType = Tensor<T, rank, options, Index>;
	
	{
		auto shape = reader.getShape();
		KJ_REQUIRE(out.rank() == shape.size());
	
		typename TensorType::Dimensions dims;
		for(size_t i = 0; i < rank; ++i) {
			if(options & Eigen::RowMajor) {
				dims[i] = shape[i];
			} else {
				dims[i] = shape[rank - i - 1];
			}
		}
	
		out.resize(dims);
	}
	
	auto data = reader.getData();
	
	KJ_REQUIRE(out.size() == data.size());
	
	auto dataOut = out.data();
	for(size_t i = 0; i < out.size(); ++i)
		dataOut[i] = (T) data[i];
}

template<typename T, typename T2>
T readTensor(T2 reader) {
	T result;
	readTensor(reader, result);
	return mv(result);
}

template<typename T, typename Reader>
Own<TensorMap<const T>> mapTensor(Reader reader) {
	using MapType    = TensorMap<const T>;
	
	using Dims = typename T::Dimensions;
	using Num  = typename T::Scalar;
	
	constexpr int rank = T::NumIndices;
	constexpr int options = T::Options;
	
	static_assert(kj::isSameType<Num, decltype(reader.getData()[0])>(), "Tensor value types must match");
	
	auto shape = reader.getShape();
	KJ_REQUIRE(shape.size() == rank);
	
	Dims dims;
	size_t size = 1;
	for(int i = 0; i < rank; ++i) {
		if(options & Eigen::RowMajor) {
			dims[i] = shape[i];
		} else {
			dims[i] = shape[rank - i - 1];
		}
		size *= dims[i];
	}
	
	auto data = reader.getData();
	
	KJ_REQUIRE(size == data.size());
		
	// Fast path for little-endian CPUs
	#ifdef FSC_WIRE_MATCHES_NATIVE
		return kj::heap<MapType>((const Num*) capnp::toAny(data).getRawBytes().begin(), dims);
	#else
		auto tensor = heapHeld<RemoveConst<T>>(dims);
		auto dataOut = tensor->data();
		for(size_t i = 0; i < data.size(); ++i)
			dataOut[i] = data[i];
		
		return kj::heap<MapType>(*tensor).attach(tensor.x());
	#endif
}
	
// template<typename T, int rank, int options, typename Index, typename T2>
template<typename TensorType, typename T2>
void writeTensor(const TensorType& in, T2 builder) {
	// using TensorType = Tensor<T, rank, options, Index>;
	constexpr int rank = TensorType::NumIndices;
	
	{
		auto shape = builder.initShape(rank);
	
		auto dims = in.dimensions();
		size_t size = 1;
		for(int i = 0; i < rank; ++i) {
			if(TensorType::Options & Eigen::RowMajor) {
				shape.set(i, dims[i]);
			} else {
				shape.set(rank - i - 1, dims[i]);
			}
		}
	}
	
	auto dataOut = builder.initData(in.size());
	auto data = in.data();
	
	KJ_REQUIRE(in.size() == dataOut.size());
	
	for(size_t i = 0; i < in.size(); ++i)
		dataOut.set(i, data[i]);
}

}