#pragma once

# include <thread>
# include <kj/debug.h>
# include <kj/memory.h>
# include <capnp/endian.h>

#include "eigen.h"
#include "kernels.h"

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	#define FSC_WIRE_MATCHES_NATIVE
#endif

namespace fsc {

constexpr double pi = 3.14159265358979323846; // "Defined" in magnetics.cpp

template<typename T, unsigned int n>
using TVec = Eigen::TensorFixedSize<T, Eigen::Sizes<n>>;

template<typename T>
using TVec3 = TVec<T, 3>;

template<typename T>
using TVec4 = TVec<T, 4>;

using TVec3d = TVec3<double>;
using TVec4d = TVec4<double>;

template<typename T, unsigned int n>
using Vec = Eigen::Vector<T, n>;

template<typename T>
using Vec2 = Vec<T, 2>;

template<typename T>
using Vec3 = Vec<T, 3>;

template<typename T>
using Vec4 = Vec<T, 4>;

using Vec3d = Vec<double, 3>;
using Vec4d = Vec<double, 4>;

using Vec3f = Vec<float, 3>;
using Vec4f = Vec<float, 4>;

using Vec3u = Vec<unsigned int, 3>;
using Vec4u = Vec<unsigned int, 4>;

using Vec3i = Vec<int, 3>;
using Vec4i = Vec<int, 4>;

template<typename T>
using Mat4 = Eigen::Matrix<T, 4, 4>;

template<typename T>
using Mat3 = Eigen::Matrix<T, 3, 3>;

using Mat4d = Mat4<double>;
using Mat3d = Mat3<double>;

namespace internal {
	template<typename LHS, typename Device>
	struct OnDeviceAssignment {
		LHS lhs;
		Device& device;
		Promise<void> prereq;
		
		bool consumed = false;
		
		OnDeviceAssignment(LHS lhs, Device& device, Promise<void> prereq) :
			lhs(lhs),
			device(device),
			prereq(mv(prereq))
		{}
		
		template<typename RHS>
		Promise<void> operator=(const RHS& rhs) {
			KJ_REQUIRE(!consumed);
			consumed = true;
			
			Device& device = this -> device;
			LHS& lhs = this -> lhs;
			
			return prereq.then([&device, lhs, rhs, prereq = mv(prereq)]() {
				auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
				auto callback = [fulfiller = mv(paf.fulfiller)]() mutable {
					fulfiller -> fulfill();
				};
				
				lhs.device(device, mv(callback)) = rhs;
				return mv(paf.promise);
			});
		}
		
		template<typename RHS>
		Promise<void> operator+=(const RHS& rhs) {
			KJ_REQUIRE(!consumed);
			consumed = true;
			
			Device& device = this -> device;
			LHS& lhs = this -> lhs;
			
			return prereq.then([&device, lhs, rhs, prereq = mv(prereq)]() {
				auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
				auto callback = [fulfiller = mv(paf.fulfiller)]() mutable {
					fulfiller -> fulfill();
				};
				
				lhs.device(device, mv(callback)) += rhs;
				return mv(paf.promise);
			});
		}
		
		template<typename RHS>
		Promise<void> operator-=(const RHS& rhs) {
			KJ_REQUIRE(!consumed);
			consumed = true;
			
			Device& device = this -> device;
			LHS& lhs = this -> lhs;
			
			return prereq.then([&device, lhs, rhs, prereq = mv(prereq)]() {
				auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
				auto callback = [fulfiller = mv(paf.fulfiller)]() mutable {
					fulfiller -> fulfill();
				};
				
				lhs.device(device, mv(callback)) -= rhs;
				return mv(paf.promise);
			});
		}
	};
}

template<typename LHS, typename Device>
Promise<void> onDevice(LHS lhs, Device& device, Promise<void> prereq = READY_NOW) {
	return internal::OnDeviceAssignment<LHS, Device>(lhs, device, prereq);
}

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
	{};

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
	{};

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
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
	{}

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
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
	{}

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
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
	{};

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
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
	{};

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
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
	{}

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
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
	{}

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	Parent get() { return Parent(*this); }
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
Own<TensorMap<T>> mapTensor(Reader reader) {
	using MapType    = TensorMap<T>;
	
	using Dims = typename T::Dimensions;
	using Num  = typename T::Scalar;
	
	constexpr int rank = T::NumIndices;
	constexpr int options = T::Options;
	
	static_assert(kj::isConst<T>(), "Can only produce const tensors");
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
	
	KJ_REQUIRE(size == data.size);
		
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
	
template<typename T, int rank, int options, typename Index, typename T2>
void writeTensor(const Tensor<T, rank, options, Index>& in, T2 builder) {
	using TensorType = Tensor<T, rank, options, Index>;
	
	{
		auto shape = builder.initShape(rank);
	
		auto dims = in.dimensions();
		size_t size = 1;
		for(size_t i = 0; i < rank; ++i) {
			if(options & Eigen::RowMajor) {
				shape.set(i, dims[i]);
			} else {
				shape.set(rank - i - 1, dims[i]);
			}
		}
	}
	
	auto dataOut = builder.initData(in.size());
	auto data = in.data();
	
	KJ_REQUIRE(in.size() == data.size());
	
	for(size_t i = 0; i < in.size(); ++i)
		dataOut.set(i, data[i]);
}

}