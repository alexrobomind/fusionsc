#pragma once

# include <thread>
# include <kj/debug.h>
# include <kj/memory.h>

#include "common.h"
#include "device-tensor.h"

namespace fsc {
	
	template<typename T, int rank, int options, typename Index, typename T2>
	void readTensor(T2 reader, Tensor<T, rank, options, Index>& out);
	
	template<typename T, typename T2>
	T readTensor(T2 reader);

	template<typename T, int rank, int options, typename Index, typename T2>
	void writeTensor(const Tensor<T, rank, options, Index>& in, T2 builder);
	
}

// Implementation

namespace fsc {
	
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
	
template<typename T, int rank, int options, typename Index, typename T2>
void writeTensor(const Tensor<T, rank, options, Index>& in, T2 builder) {
	using TensorType = Tensor<T, rank, options, Index>;
	
	{
		auto shape = builder.initShape(rank);
	
		auto dims = in.dimensions();
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
	for(size_t i = 0; i < in.size(); ++i)
		dataOut.set(i, data[i]);
}

}