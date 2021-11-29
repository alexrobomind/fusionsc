#pragma once

# define EIGEN_USE_THREADS 1

# include <unsupported/Eigen/CXX11/Tensor>
# include <unsupported/Eigen/CXX11/ThreadPool>
# include <Eigen/Dense>
# include <cmath>
# include <thread>

# ifdef _OPENMP
	# include <omp.h>
# endif

#define FSC_OFFLOAD 0

namespace fsc {
	using Eigen::Tensor;
	using Eigen::TensorFixedSize;
	using Eigen::TensorRef;
	using Eigen::TensorMap;
	using Eigen::Sizes;
	
	template<typename T>
	using Vec3 = TensorFixedSize<T, Sizes<3>>;
	
	template<typename T>
	typename T::Scalar normSq(const T& t) { TensorFixedSize<typename T::Scalar, Sizes<>> result = t.square().sum(); return result(); }
	
	template<typename T>
	typename T::Scalar norm(const T& t) { return sqrt(normSq(t)); }
	
	template<typename T1, typename T2>
	Vec3<typename T1::Scalar> cross(const T1& t1, const T2& t2);
	
	template<typename T, typename Device>
	struct MappedTensor { static_assert(sizeof(T) == 0, "Mapper not implemented"); };

	template<typename TVal, int rank, int options, typename Index, typename Device>
	struct MappedTensor<Tensor<TVal, rank, options, Index>, Device>;

	template<typename TVal, typename Dims, int options, typename Index, typename Device>
	struct MappedTensor<TensorFixedSize<TVal, Dims, options, Index>, Device>;
	
	template<typename T, typename Device>
	struct MappedData { static_assert(sizeof(T) == 0, "Mapper not implemented"); };
	
	template<typename T>
	struct MappedData<T, Eigen::ThreadPoolDevice>;
	
	// Number of threads to be used for evaluation
	inline unsigned int numThreads() {
		# ifdef _OPENMP
			return omp_get_max_threads();
		# else
			return std::thread::hardware_concurrency();
		# endif
	}
	
	/*class OffloadDevice {
	public:
		Eigen::ThreadPoolDevice& eigenDevice() { return _eigenDevice; }
		
		OffloadDevice() :
			_tp(numThreads()),
			_eigenDevice(&_tp, numThreads())
		{}
		
	private:
		Eigen::ThreadPool _tp;
		Eigen::ThreadPoolDevice _eigenDevice;
	};*/
}

// Implementation

namespace fsc {

template<typename T1, typename T2>
Vec3<typename T1::Scalar> cross(const T1& t1, const T2& t2) {
	using Num = typename T1::Scalar;
	
	Vec3<Num> r1 = t1;
	Vec3<Num> r2 = t2;
	
	Vec3<Num> result;
	result(0) = r1(1) * r2(2) - r2(1) * r1(2);
	result(1) = r1(2) * r2(0) - r2(2) * r1(0);
	result(2) = r1(0) * r2(1) - r2(0) * r1(1);
	
	return result;
}

template<typename T>
struct MappedData<T, Eigen::ThreadPoolDevice> {
	T* ptr;
	
	MappedData(Eigen::ThreadPoolDevice& device, T* ptr, size_t size) : ptr(ptr) {}
	static T* deviceAlloc(Eigen::ThreadPoolDevice& device, T* ref, size_t size) { return ref; }
	
	T* devicePtr() { return ptr; }
	void updateHost() {}
	void updateDevice() {}
};

template<typename TVal, int tRank, int tOpts, typename Index, typename Device>
struct MappedTensor<Tensor<TVal, tRank, tOpts, Index>, Device> : public TensorMap<Tensor<TVal, tRank, tOpts, Index>> {
	using Maps = Tensor<TVal, tRank, tOpts, Index>;

	MappedData<TVal, Device> _data;

	MappedTensor(Maps& target, Device& device) :
		TensorMap<Tensor<TVal, tRank, tOpts, Index>>(
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size()),
			target.dimensions()
		),
		_data(device, target.data(), target.size())
	{};

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
};

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MappedTensor<TensorFixedSize<TVal, Dims, options, Index>, Device> : public TensorMap<TensorFixedSize<TVal, Dims, options, Index>> {
	using Maps = TensorFixedSize<TVal, Dims, options, Index>;
	
	MappedData<TVal, Device> _data;

	MappedTensor(Maps& target, Device& device) :
		TensorMap<TensorFixedSize<TVal, Dims, options, Index>> (
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size())
		),
		_data(target.data(), target.size())
	{}

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
};
	
}
