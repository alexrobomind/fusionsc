#pragma once

#include "common.h"
#include <H5Cpp.h>

namespace fsc {
	template<typename T>
	const H5::PredType& h5Type();
	
	struct H5Dim {
		hsize_t length;
		hsize_t maxLength;
		Maybe<H5::DataSet> dataset;
		
		H5Dim(const H5Dim& other) = default;
		
		/** Dimension scale constructor
		 *
		 * Extracts the information from the target dimension scale object. Requires the target dataset to begin
		 * a 1D dataset (NetCDF style)
		 */
		H5Dim(const H5::DataSet&);
		
		inline H5Dim(const H5::DataSet& ds, hsize_t len, hsize_t maxLen) : length(len), maxLength(maxLen), dataset(ds) {}
		inline H5Dim(hsize_t len, hsize_t maxLen) : length(len), maxLength(maxLen) {}
		inline H5Dim(hsize_t len) : H5Dim(len, len) {}
		
		static inline H5Dim unlimited(hsize_t length) { return H5Dim(length, H5S_UNLIMITED); }
	};
	
	H5::DataSet createDataSet(H5::H5Location&, kj::StringPtr, const H5::DataType&, kj::ArrayPtr<const H5Dim> = nullptr);
	H5::DataSet createDimension(H5::H5Location&, kj::StringPtr, const H5::DataType&, const H5Dim&);
	
	kj::Array<H5Dim> getDimensions(const H5::DataSet&);
	size_t totalSize(const H5::DataSpace&);
	
	template<typename Reader>
	kj::Array<H5Dim> h5TensorDims(Reader reader);
	
	template<typename T>
	H5::DataSet createDataSet(H5::H5Location&, kj::StringPtr, kj::ArrayPtr<const H5Dim> = nullptr);
	
	template<typename T>
	H5::DataSet createDimension(H5::H5Location&, kj::StringPtr, const H5Dim&);
	
	template<typename T>
	T readScalar(const H5::DataSet&);
	
	template<typename T>
	void writeScalar(const H5::DataSet&, T);
	
	template<typename T>
	Array<T> readArray(const H5::DataSet&);
	
	template<typename T>
	void writeArray(const H5::DataSet&, const kj::ArrayPtr<const T>& data);
	
	template<typename Builder, typename T = capnp::FromBuilder<Builder>>
	void readTensor(const H5::DataSet&, Builder);
	
	template<typename Reader, typename T = capnp::FromAny<Reader>>
	void writeTensor(const H5::DataSet&, Reader);
}

// Inline implementation

namespace fsc {
	namespace internal {
		template<typename T>
		struct H5TypeFor {
			static_assert(sizeof(T) == 0, "No native type assigned");
		};
		
		#define FSC_HDF5_NATIVE_TYPE(cType, hdfType) \
		template<> \
		struct H5TypeFor<cType> { \
			static const inline H5::PredType& value = H5::PredType::hdfType; \
		};
		
		// The types char, unsigned char (uint8_t), and signed char (int8_t) are MUTUALLY DISTINCT
		FSC_HDF5_NATIVE_TYPE(char, NATIVE_CHAR);
		FSC_HDF5_NATIVE_TYPE(int8_t, NATIVE_INT8);
		FSC_HDF5_NATIVE_TYPE(uint8_t, NATIVE_UINT8);
		
		FSC_HDF5_NATIVE_TYPE(uint16_t, NATIVE_UINT16);
		FSC_HDF5_NATIVE_TYPE(uint32_t, NATIVE_UINT32);
		FSC_HDF5_NATIVE_TYPE(uint64_t, NATIVE_UINT64);
		
		FSC_HDF5_NATIVE_TYPE(int16_t, NATIVE_INT16);
		FSC_HDF5_NATIVE_TYPE(int32_t, NATIVE_INT32);
		FSC_HDF5_NATIVE_TYPE(int64_t, NATIVE_INT64);
		
		FSC_HDF5_NATIVE_TYPE(float, NATIVE_FLOAT);
		FSC_HDF5_NATIVE_TYPE(double, NATIVE_DOUBLE);
		
		#undef FSC_HDF5_NATIVE_TYPE
	}
	
	template<typename T>
	const H5::PredType& h5Type() {
		return internal::H5TypeFor<T>::value;
	}
	
	template<typename Reader>
	kj::Array<H5Dim> h5TensorDims(Reader reader) {
		auto shape = reader.getShape();
		auto result = kj::heapArrayBuilder<H5Dim>(shape.size());
		for(auto i : kj::indices(shape)) {
			result.add(shape[i]);
		}
		return result.finish();
	}
	
	template<typename T>
	H5::DataSet createDataSet(H5::H5Location& loc, kj::StringPtr name, kj::ArrayPtr<const H5Dim> dims) {
		return createDataSet(loc, name, h5Type<T>(), dims);
	}
	
	template<typename T>
	H5::DataSet createDimension(H5::H5Location& loc, kj::StringPtr name, const H5Dim& dim) {
		return createDimension(loc, name, h5Type<T>(), dim);
	}
	
	template<typename T>
	T readScalar(const H5::DataSet& ds) {
		// KJ_REQUIRE(totalSize(ds.getSpace) == 1);
		
		T buffer;
		ds.read(&buffer, h5Type<T>(), H5::DataSpace());
		
		return buffer;
	}
	
	template<typename T>
	void writeScalar(const H5::DataSet& ds, T val) {
		ds.write(&val, h5Type<T>(), H5::DataSpace());
	}
	
	template<typename T>
	Array<T> readArray(const H5::DataSet& ds) {
		auto result = kj::heapArray<T>(totalSize(ds.getSpace()));
		ds.read(result.begin(), h5Type<T>());
		return result;
	}
	
	template<typename T>
	void writeArray(const H5::DataSet& ds, const ArrayPtr<const T>& data) {
		KJ_REQUIRE(totalSize(ds.getSpace()) == data.size());
		
		ds.write(data.begin(), h5Type<T>());
	}
	
	template<typename T, typename Builder>
	void readTensor(const H5::DataSet& ds, Builder out) {
		auto dims = getDimensions(ds);
		auto shape = out.initShape(dims.size());
		size_t shapeProd = 1;
		for(auto i : kj::indices(dims)) {
			size_t thisDim = dims[i].length;
			shape.set(i, thisDim);
			shapeProd *= thisDim;
		}
		
		auto fileData = readArray<decltype(out.getData()[0])>(ds);
		auto data = out.initData(fileData.size());
		
		for(auto i : kj::indices(data)) {
			data.set(i, fileData[i]);
		}
	}
	
	template<typename T, typename Reader>
	void writeTensor(const H5::DataSet& ds, Reader in) {
		auto data = in.getData();
		auto tmp = kj::heapArray<decltype(data[0])>(data.size());
		for(auto i : kj::indices(data))
			tmp[i] = data[i];
		
		writeArray(ds, data.asPtr());
	}
}