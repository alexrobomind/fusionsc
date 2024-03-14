#pragma once

# include <thread>
# include <kj/debug.h>
# include <kj/memory.h>
# include <capnp/endian.h>

#include "eigen.h"
#include "memory.h"

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	#define FSC_WIRE_MATCHES_NATIVE
#endif

namespace fsc {

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

template<typename T, int rank, int options, typename Index, typename T2>
void readVardimTensor(T2 reader, size_t variableDim, Tensor<T, rank, options, Index>& out) {
	using TensorType = Tensor<T, rank, options, Index>;
	
	{
		auto shape = reader.getShape();
		KJ_REQUIRE(rank <= shape.size() + 1);
		KJ_REQUIRE(variableDim < rank);
		
		size_t variableDimSize = 1;
		for(auto iDim : kj::range(variableDim, variableDim + 1 + shape.size() - rank))
			variableDimSize *= shape[iDim];
	
		typename TensorType::Dimensions dims;
		for(size_t i = 0; i < rank; ++i) {
			size_t entry;
			if(i < variableDim)
				entry = shape[i];
			else if(i == variableDim)
				entry = variableDimSize;
			else
				entry = shape[i + shape.size() - rank];
			
			if(options & Eigen::RowMajor) {
				dims[i] = entry;
			} else {
				dims[rank - i - 1] = entry;
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
		auto rawBytes = capnp::toAny(data).getRawBytes();
		// Check if data is densely packed
		if(rawBytes.size() == size * sizeof(T)) {
			return kj::heap<MapType>((const Num*) rawBytes.begin(), dims);
		}
	#endif
	
	auto tensor = heapHeld<RemoveConst<T>>(dims);
	auto dataOut = tensor->data();
	for(size_t i = 0; i < data.size(); ++i)
		dataOut[i] = data[i];
	
	return kj::heap<MapType>(*tensor).attach(tensor.x());
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

template<typename T>
void validateTensor(T t, kj::ArrayPtr<const Maybe<uint32_t>> shapeConstraints = nullptr) {
	auto shape = t.getShape();
	
	uint32_t shapeProd = 1;
	for(auto el : shape)
		shapeProd *= el;
	
	KJ_REQUIRE(t.getData().size() == shapeProd, "Data size does not match expectation from shape");
	
	auto desiredRank = shapeConstraints.size();
	if(desiredRank != 0) {
		KJ_REQUIRE(shape.size() == desiredRank, "Rank mismatch");
	}
	
	for(auto dimension : kj::range(0, desiredRank)) {
		KJ_IF_MAYBE(pExp, shapeConstraints[dimension]) {
			uint32_t expectedSize = *pExp;
			KJ_REQUIRE(shape[dimension] == expectedSize, "Shape mismatch along dimension", dimension);
		}
	}
}

}