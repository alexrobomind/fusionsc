#pragma once

# include <thread>
# include <kj/debug.h>
# include <kj/memory.h>
# include <capnp/endian.h>

#include "eigen.h"
#include "kernels.h"
#include "device.h"

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	#define FSC_WIRE_MATCHES_NATIVE
#endif

namespace fsc {

template<TensorType>
struct TensorMapping<TensorType> : public DeviceMapping<kj::Array<typename TensorType::Scalar>> {
	using Scalar = typename TensorType::Scalar;
	
	TensorMap<TensorType> hostMap;
	TensorMap<TensorType> deviceMap;
	
	DeviceMapping(Held<TensorType> tensor, DeviceBase& device) :
		DeviceMaping<kj::Array<TVal>>(kj::ArrayPtr<Scalar>(tensor -> data(), tensor -> size()).attach(tensor.x())),
		hostMap(*tensor),
		deviceMap(DeviceMapping<kj::Array<typename TensorType::Scalar>>::get(), tensor -> dimensions())
	{}
	
	TensorMap<TensorType> get() { return deviceMap; }
};


// ------------------- Lots of different device mappings for tensors -------------------------

template<typename TVal, int tRank, int tOpts, typename Index, typename Device>
struct MapToDevice<Tensor<TVal, tRank, tOpts, Index>, Device> : public TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>> {
	using MapsFrom = Tensor<TVal, tRank, tOpts, Index>;
	using Parent = TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>>;

	MappedData<TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.size()),
			target.dimensions()
		),
		_data(device, target.data() /* Host pointer */, this->data() /* Device pointer allocated above */, this->size() /* Same as target.size() */)
	{}

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
};

template<typename TVal, int tRank, int tOpts, typename Index, typename Device>
struct MapToDevice<const Tensor<TVal, tRank, tOpts, Index>, Device> : public TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>> {
	using MapsFrom = const Tensor<TVal, tRank, tOpts, Index>;
	using Parent = TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>>;

	MappedData<const TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.size()),
			target.dimensions()
		),
		_data(device, target.data() /* Host pointer */, this->data() /* Device pointer allocated above */, this->size() /* Same as target.size() */)
	{ /*KJ_DBG(target.data(), this->data());*/ }

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { KJ_DBG("get()", this->data()); return Parent(*this); }
};

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MapToDevice<TensorFixedSize<TVal, Dims, options, Index>, Device> : public TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>> {
	using MapsFrom = TensorFixedSize<TVal, Dims, options, Index>;
	using Parent = TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>>;
	
	MappedData<TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size())
		),
		_data(target.data(), target.size())
	{ /*KJ_DBG(target.data(), this->data());*/ }

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { KJ_DBG("get()", this->data()); return Parent(*this); }
};

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MapToDevice<const TensorFixedSize<TVal, Dims, options, Index>, Device> : public TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>> {
	using MapsFrom = TensorFixedSize<TVal, Dims, options, Index>;
	using Parent = TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>>;
	
	MappedData<const TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size())
		),
		_data(target.data(), target.size())
	{ /*KJ_DBG(target.data(), this->data());*/ }

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { /*KJ_DBG("get()", this->data());*/ return Parent(*this); }
};

template<typename TVal, int tRank, int tOpts, typename Index, typename Device>
struct MapToDevice<TensorMap<Tensor<TVal, tRank, tOpts, Index>>, Device> : public TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>> {
	using MapsFrom = TensorMap<Tensor<TVal, tRank, tOpts, Index>>;
	using Parent = TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>>;

	MappedData<TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.size()),
			target.dimensions()
		),
		_data(device, target.data() /* Host pointer */, this->data() /* Device pointer allocated above */, this->size() /* Same as target.size() */)
	{ /*KJ_DBG(target.data(), this->data());*/ }

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { /*KJ_DBG("get()", this->data());*/ return Parent(*this); }
};

template<typename TVal, int tRank, int tOpts, typename Index, typename Device>
struct MapToDevice<TensorMap<const Tensor<TVal, tRank, tOpts, Index>>, Device> : public TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>> {
	using MapsFrom = TensorMap<const Tensor<TVal, tRank, tOpts, Index>>;
	using Parent = TensorMap<Tensor<RemoveConst<TVal>, tRank, tOpts, Index>>;

	MappedData<const TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.size()),
			target.dimensions()
		),
		_data(device, target.data() /* Host pointer */, this->data() /* Device pointer allocated above */, this->size() /* Same as target.size() */)
	{ /*KJ_DBG(target.data(), this->data());*/ }

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { /*KJ_DBG("get()", this->data());*/ return Parent(*this); }
};

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MapToDevice<TensorMap<TensorFixedSize<TVal, Dims, options, Index>>, Device> : public TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>> {
	using MapsFrom = TensorMap<TensorFixedSize<TVal, Dims, options, Index>>;
	using Parent = TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>>;
	
	MappedData<TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size())
		),
		_data(target.data(), target.size())
	{ /*KJ_DBG(target.data(), this->data());*/ }

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { KJ_DBG("get()", this->data()); return Parent(*this); }
};

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MapToDevice<TensorMap<const TensorFixedSize<TVal, Dims, options, Index>>, Device> : public TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>> {
	using MapsFrom = TensorMap<const TensorFixedSize<TVal, Dims, options, Index>>;
	using Parent = TensorMap<TensorFixedSize<RemoveConst<TVal>, Dims, options, Index>>;
	
	MappedData<const TVal, Device> _data;

	MapToDevice(MapsFrom& target, Device& device) :
		Parent(
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size())
		),
		_data(target.data(), target.size())
	{ /*KJ_DBG(target.data(), this->data());*/ }

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { KJ_DBG("get()", this->data()); return Parent(*this); }
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