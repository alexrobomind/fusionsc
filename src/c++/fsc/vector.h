#pragma once

#include "tensor.h"
#include <cmath>

namespace fsc {

template<typename T, unsigned int n>
struct Vec {
	T data[n];
	
	EIGEN_DEVICE_FUNC Vec() = default;
	
	EIGEN_DEVICE_FUNC Vec(T x) {
		for(unsigned int i = 0; i < n; ++i)
			data[i] = x;
	}
	
	T& EIGEN_DEVICE_FUNC operator[](unsigned int i) {
		return data[i];
	}
	
	T& EIGEN_DEVICE_FUNC operator()(unsigned int i) {
		return data[i];
	}
	
	const T& EIGEN_DEVICE_FUNC operator[](unsigned int i) const {
		return data[i];
	}
	
	const T& EIGEN_DEVICE_FUNC operator()(unsigned int i) const {
		return data[i];
	}
	
	auto EIGEN_DEVICE_FUNC operator*(const T fac) const {
		Vec<T, n> result;
		for(unsigned int i = 0; i < n; ++i)
			result[i] = data[i] * fac;
		
		return result;
	}
	
	auto EIGEN_DEVICE_FUNC operator/(const T fac) const {
		Vec<T, n> result;
		for(unsigned int i = 0; i < n; ++i)
			result[i] = data[i] / fac;
		
		return result;
	}
	
	Vec<T, n>& EIGEN_DEVICE_FUNC operator*=(const T fac) {
		for(unsigned int i = 0; i < n; ++i)
			data[i] *= fac;
		return *this;
	}
	
	Vec<T, n>& EIGEN_DEVICE_FUNC operator/=(const T fac) {
		for(unsigned int i = 0; i < n; ++i)
			data[i] /= fac;
		return *this;
	}
	
	template<typename T2>
	auto EIGEN_DEVICE_FUNC operator+(const Vec<T2, n>& other) const  {
		using T3 = decltype(other.data[0] + data[0]);
		Vec<T3, n> result;
		
		for(unsigned int i = 0; i < n; ++i)
			result[i] = data[i] + other.data[i];
		
		return result;
	}
	
	template<typename T2>
	auto EIGEN_DEVICE_FUNC operator-(const Vec<T2, n>& other) const  {
		using T3 = decltype(other.data[0] - data[0]);
		Vec<T3, n> result;
		
		for(unsigned int i = 0; i < n; ++i)
			result[i] = data[i] - other.data[i];
		
		return result;
	}
	
	template<typename T2>
	auto EIGEN_DEVICE_FUNC operator*(const Vec<T2, n>& other) const  {
		using T3 = decltype(other.data[0] * data[0]);
		Vec<T3, n> result;
		
		for(unsigned int i = 0; i < n; ++i)
			result[i] = data[i] * other.data[i];
		
		return result;
	}
	
	template<typename T2>
	auto EIGEN_DEVICE_FUNC operator/(const Vec<T2, n>& other) const {
		using T3 = decltype(other.data[0] / data[0]);
		Vec<T3, n> result;
		
		for(unsigned int i = 0; i < n; ++i)
			result[i] = data[i] / other.data[i];
		
		return result;
	}
	
	template<typename T2>
	Vec<T, n>& EIGEN_DEVICE_FUNC operator+=(const Vec<T2, n>& other) {
		for(unsigned int i = 0; i < n; ++i)
			data[i] += other.data[i];
		return *this;
	}
	
	template<typename T2>
	Vec<T, n>& EIGEN_DEVICE_FUNC operator-=(const Vec<T2, n>& other) {
		for(unsigned int i = 0; i < n; ++i)
			data[i] -= other.data[i];
		return *this;
	}
	
	template<typename T2>
	Vec<T, n>& EIGEN_DEVICE_FUNC operator*=(const Vec<T2, n>& other) {
		for(unsigned int i = 0; i < n; ++i)
			data[i] *= other.data[i];
		return *this;
	}
	
	template<typename T2>
	Vec<T, n>& EIGEN_DEVICE_FUNC operator/=(const Vec<T2, n>& other) {
		for(unsigned int i = 0; i < n; ++i)
			data[i] /= other.data[i];
		return *this;
	}
	
	T EIGEN_DEVICE_FUNC normSq() const {
		T result = 0;
		
		for(unsigned int i = 0; i < n; ++i)
			result += data[i] * data[i];
		
		return result;
	}
	
	auto EIGEN_DEVICE_FUNC norm() const {
		return sqrt(normSq());
	}
};

template<typename T, unsigned int n>
auto EIGEN_DEVICE_FUNC operator*(T t, const Vec<T, n>& other) {
	return other * t;
}

template<typename T>
using Vec4 = Vec<T, 4>;

template<typename T>
using Vec3 = Vec<T, 3>;

template<typename T>
using Vec2 = Vec<T, 2>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;

using Vec2i = Vec2<int>;
using Vec3i = Vec3<int>;
using Vec4i = Vec4<int>;

using Vec2u = Vec2<unsigned int>;
using Vec3u = Vec3<unsigned int>;
using Vec4u = Vec4<unsigned int>;

template<typename T>
Vec3<T> EIGEN_DEVICE_FUNC cross(const Vec3<T>& t1, const Vec3<T>& t2) {	
	Vec3<T> result;
	result(0) = t1(1) * t2(2) - t2(1) * t1(2);
	result(1) = t1(2) * t2(0) - t2(2) * t1(0);
	result(2) = t1(0) * t2(1) - t2(0) * t1(1);
	
	return result;
}

}

