#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <cmath>

#define FSC_OFFLOAD 0

namespace fsc {
	using Eigen::Tensor;
	using Eigen::TensorFixedSize;
	using Eigen::TensorRef;
	using Eigen::Sizes;
	
	template<typename T>
	using Vec3 = TensorFixedSize<T, Sizes<3>>;
	
	template<typename T>
	typename T::Scalar normSq(const T& t) { return t.square().sum(); }
	
	template<typename T>
	typename T::Scalar norm(const T& t) { return sqrt(norm(t)); }
	
	template<typename T1, typename T2>
	Vec3<typename T1::Scalar> cross(const T1& t1, const T2& t2);
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

}