#pragma once

#include <fsc/magnetics.capnp.cu.h>
#include <fsc/magnetics.capnp.h>

#include "tensor.h"
#include "grids.h"

namespace fsc {

/**
 * Simple fast linear interpolation strategy
 */
template<typename Num>
struct LinearInterpolation {
	using Scalar = Num;
	using Coeffs = std::array<Scalar, 2>;
	
	constexpr EIGEN_DEVICE_FUNC size_t nPoints() { return 2; }
	constexpr EIGEN_DEVICE_FUNC std::array<Scalar, 2> coefficients(Num x) {
		return {1 - x, x};
	}
	constexpr EIGEN_DEVICE_FUNC std::array<int, 2> offsets() {
		return {0, 1};
	}
};

/**
 * Continuously differentiable interpolation based on 3rd order Hermite splines.
 * Interpolates exact up to 2nd order. Values on interval endpoints are exact. Derivatives
 * equal their central finite difference.
 */
template<typename Num>
struct C1CubicInterpolation {
	// Cubic interpolation based on Hermite splines.
	// Performs polynomial interpolation on [0, 1] with constraints on f(0), f'(0), f(1), and f'(1)
	// Derivatives are derived as f'(0) = 0.5 * (f(1) - f(-1)), f'(1) = 0.5 * (f(2) - f(0))
	//
	// Implementation notes:
	// The Hermite polynomial is given as P(x) = SUM_i:0 -> n f[x0, ..., xi] PROD_j:0 -> i-1 (x - xj)
	//
	// We have (x0, x1, x2, x3) = (0, 0, 1, 1)
	//
	// This gives us:
	//
	// f[x0] = f(0)
	// f[x0, x1] = f'(0)
	//
	// f[x0, x1, x2] = f[x1, x2] - f[x0, x1]
	//               = f[x2] - f[x1] - f[x0, x1]
    //               = f(1) - f(0) - f'(0)
	//
	// f[x0, x1, x2, x3] = f[x1, x2, x3] - f[x0, x1, x2]
	//                   = f[x2, x3] - f[x1, x2] - f[x0, x1, x2]
	//                   = f[x2, x3] - (f[x2] - f[x1]) - f[x0, x1, x2]
	//                   = f'(1) - f(1) + f(0) - f[x0, x1, x2]
	
	// For the different polynomials we get:
	// - Associated with f(0):
	//   f[x0] = 1
	//   f[x0, x1] = 0
	//   f[x0, ..., x2] = -1
	//   f[x0, ..., x3] = 2
	//    ==> P[x] = 1 - x**2 + 2 * x**2 * (x - 1) [Check]
	// - Associated with f(1):
	//   f[x0] = 0
	//   f[x0, x1] = 0
	//   f[x0, ..., x2] = 1
	//   f[x0, ..., x3] = -2
	//    ==> P[x] = x**2 - 2 * x**2 * (x - 1) [Check]
	// - Associated with f'(0):
	//   f[x0] = 0
	//   f[x0, x1] = 1
	//   f[x0, ..., x2] = -1
	//   f[x0, ..., x3] = 1
	//    ==> P[x] = x - x**2 + x**2 * (x - 1) [Check]
	// - Associated with f'(1):
	//   f[x0] = 0
	//   f[x0, x1] = 0
	//   f[x0, ..., x2] = 0
	//   f[x0, ..., x3] = 1
	//    ==> P[x] = x**2 * (x - 1) [Check]
	
	using Scalar = Num;
	using Coeffs = std::array<Scalar, 4>;
	
	constexpr EIGEN_DEVICE_FUNC size_t nPoints() { return 4; }
	constexpr EIGEN_DEVICE_FUNC std::array<Scalar, 4> coefficients(Num x) {		
		Num pf0 = 1 - x*x + 2 * (x*x) * (x-1);
		Num pf1 = x*x - 2 * x*x * (x-1);
		Num pdf0 = x - (x*x) + x*x * (x-1);
		Num pdf1 = x*x * (x-1);
		
		return {-0.5 * pdf0, pf0 - 0.5 * pdf1, pf1 + 0.5 * pdf0, 0.5 * pdf1};
	}
	constexpr EIGEN_DEVICE_FUNC std::array<int, 4> offsets() {
		return {-1, 0, 1, 2};
	}
};

template<typename Strategy, typename Vec, size_t... indices>
std::array<typename Strategy::Coeffs, sizeof...(indices)> calculateCoeffsND(Strategy& strategy, const Vec& lx, std::index_sequence<indices...> indexSequence) {
	return { strategy.coefficients(lx[indices]) ... };
}

template<int nDims, typename Strategy, int iDim>
struct NDInterpEvaluator {
	using Scalar = typename Strategy::Scalar;
	
	static_assert(iDim < nDims);
	static_assert(iDim >= 0);
	
	/*const Vec<Scalar, nDims>& lx,*/
	
	template<typename F, typename... Indices>
	static EIGEN_DEVICE_FUNC Scalar evaluate(Strategy& strategy, const F& f, const Vec<int, nDims>& base, const std::array<typename Strategy::Coeffs, nDims>& coeffs, Indices... indices) {
		static_assert(sizeof...(indices) == iDim);
		
		Scalar result = 0;
		
		// auto coefficients = strategy.coefficients(lx[iDim]);
		auto& coefficients = coeffs[iDim];
		auto offsets      = strategy.offsets();
					
		for(int idx = 0; idx < strategy.nPoints(); ++idx) {
			result += coefficients[idx] * NDInterpEvaluator<nDims, Strategy, iDim + 1>::evaluate(strategy, f, base, /*lx,*/coeffs, indices..., base[iDim] + offsets[idx]);
		}
		
		return result;
	}
};

template<int nDims, typename Strategy>
struct NDInterpEvaluator<nDims, Strategy, nDims> {
	using Scalar = typename Strategy::Scalar;
	
	template<typename F, typename... Indices>
	static EIGEN_DEVICE_FUNC Scalar evaluate(Strategy& strategy, const F& f, const Vec<int, nDims>& base, /*const Vec<Scalar, nDims>& lx*/const std::array<typename Strategy::Coeffs, nDims>& coeffs, Indices... indices) {
		static_assert(sizeof...(indices) == nDims);
		
		return f(indices...);
	}
};

/** Helper that extracts the scalar value of an AutoDiff scalar */
template<typename T>
struct ValueExtractor {
	using Scalar = T;
	
	static T extract(T in) { return in; }
};

template<typename T>
struct ValueExtractor<Eigen::AutoDiffScalar<T>> {
	using Scalar = typename T::Scalar;
	
	static Scalar extract(Eigen::AutoDiffScalar<T> in) { return in.value(); }
};

/**
 * Multi-dimensional interpolator that runs based on a given
 * 1-dimensional interpolation strategy
 */
template<int nDims, typename Strategy>
struct NDInterpolator {
	using Scalar = typename Strategy::Scalar;
	static_assert(nDims >= 1);
	
	struct Axis {
		Scalar x1;
		Scalar x2;
		int nIntervals;
		
		inline EIGEN_DEVICE_FUNC Axis(Scalar x1, Scalar x2, int nIntervals) :
			x1(x1), x2(x2), nIntervals(nIntervals)
		{}
	};
	
	Strategy strategy;
	
	Scalar scaleMultipliers[nDims];
	Scalar offsets[nDims];
	
	EIGEN_DEVICE_FUNC NDInterpolator(const Strategy& strategy, const Axis axes[nDims]) :
		strategy(strategy)
	{
		for(int i = 0; i < nDims; ++i) {
			offsets[i] = -axes[i].x1;
			scaleMultipliers[i] = 1.0 / (axes[i].x2 - axes[i].x1) * axes[i].nIntervals;
		}
	}
	
	EIGEN_DEVICE_FUNC NDInterpolator(const Strategy& strategy, std::initializer_list<Axis> axes) :
		NDInterpolator(strategy, axes.begin())
	{
		CUPNP_REQUIRE(axes.size() == nDims);
	}
	
	template<typename F>
	EIGEN_DEVICE_FUNC Scalar operator()(const F& f, const Vec<Scalar, nDims>& x) {
		Vec<int, nDims> base;
		Vec<Scalar, nDims> lx;
		
		for(int i = 0; i < nDims; ++i) {
			Scalar scaled = scaleMultipliers[i] * (x[i] + offsets[i]);
			base[i] = floor(ValueExtractor<Scalar>::extract(scaled));
			lx[i] = scaled - base[i];
		}
		
		std::array<typename Strategy::Coeffs, nDims> coeffs =
			calculateCoeffsND(strategy, lx, std::make_index_sequence<nDims>());
		
		return NDInterpEvaluator<nDims, Strategy, 0>::evaluate(strategy, f, base, coeffs);
	}
};

template<typename Strategy>
struct SlabFieldInterpolator {
	using Scalar = typename Strategy::Scalar;
	
	NDInterpolator<3, Strategy> interpolator;
	using Axis = typename NDInterpolator<3, Strategy>::Axis;
	
	EIGEN_DEVICE_FUNC SlabFieldInterpolator(const Strategy& strategy, const cu::ToroidalGrid grid) :
		interpolator(strategy, {
			Axis(0, 2 * fsc::pi / grid.getNSym(), grid.getNPhi()),
			Axis(grid.getZMin(), grid.getZMax(), grid.getNZ() - 1),
			Axis(grid.getRMin(), grid.getRMax(), grid.getNR() - 1),
		})
	{
	}
	
	EIGEN_DEVICE_FUNC SlabFieldInterpolator(const Strategy& strategy, const ToroidalGridStruct& grid) :
		interpolator(strategy, {
			Axis(0, 2 * fsc::pi / grid.nSym, grid.nPhi),
			Axis(grid.zMin, grid.zMax, grid.nZ - 1),
			Axis(grid.rMin, grid.rMax, grid.nZ - 1),
		})
	{
	}
	
	// Host-only interpolator
	SlabFieldInterpolator(const Strategy& strategy, ToroidalGrid::Reader grid) :
		interpolator(strategy, {
			Axis(0, 2 * fsc::pi / grid.getNSym(), grid.getNPhi()),
			Axis(grid.getZMin(), grid.getZMax(), grid.getNZ() - 1),
			Axis(grid.getRMin(), grid.getRMax(), grid.getNR() - 1),
		})
	{
	}
	
	EIGEN_DEVICE_FUNC Vec<Scalar, 3> operator()(const TensorMap<const Tensor<Scalar, 4>>& fieldData, const Vec<Scalar, 3>& xyz) {
		Scalar x = xyz[0];
		Scalar y = xyz[1];
		Scalar z = xyz[2];
		
		Scalar r = std::sqrt(x*x + y*y);
		Scalar phi = atan2(y, x);
		
		// Lambda functions that return individual phi components
		// and clamp the field for evaluation
		auto selectComponent = [&fieldData](int iDim) {
			return [&fieldData, iDim](int iPhi, int iZ, int iR) {
				if(iR < 0) iR = 0;
				if(iZ < 0) iZ = 0;
				if(iR >= fieldData.dimension(1)) iR = fieldData.dimension(1) - 1;
				if(iZ >= fieldData.dimension(2)) iZ = fieldData.dimension(2) - 1;
				
				int nPhi = fieldData.dimension(3);
				iPhi = (iPhi % nPhi + nPhi) % nPhi;
				
				return fieldData(iDim, iR, iZ, iPhi);
			};
		};
		
		Scalar bPhi = interpolator(selectComponent(0), {phi, z, r});
		Scalar bZ   = interpolator(selectComponent(1), {phi, z, r});
		Scalar bR   = interpolator(selectComponent(2), {phi, z, r});
		
		// KJ_DBG(bPhi, bZ, bR);
		
		Vec2<Scalar> eR   = { cos(phi), sin(phi)};
		Vec2<Scalar> ePhi = {-sin(phi), cos(phi)};

		return {
			bR * eR[0] + bPhi * ePhi[0],
			bR * eR[1] + bPhi * ePhi[1],
			bZ
		};
	}
	
	EIGEN_DEVICE_FUNC Vec<Scalar, 3> inSlabOrientation(const TensorMap<const Tensor<Scalar, 4>>& fieldData, const Vec<Scalar, 3>& xyz) {
		Scalar x = xyz[0];
		Scalar y = xyz[1];
		Scalar z = xyz[2];
		
		Scalar r = std::sqrt(x*x + y*y);
		Scalar phi = atan2(y, x);
		
		// Lambda functions that return individual phi components
		// and clamp the field for evaluation
		auto selectComponent = [&fieldData](int iDim) {
			return [&fieldData, iDim](int iPhi, int iZ, int iR) {
				if(iR < 0) iR = 0;
				if(iZ < 0) iZ = 0;
				if(iR >= fieldData.dimension(1)) iR = fieldData.dimension(1) - 1;
				if(iZ >= fieldData.dimension(2)) iZ = fieldData.dimension(2) - 1;
				
				int nPhi = fieldData.dimension(3);
				iPhi = (iPhi % nPhi + nPhi) % nPhi;
				
				return fieldData(iDim, iR, iZ, iPhi);
			};
		};
		
		Scalar bPhi = interpolator(selectComponent(0), {phi, z, r});
		Scalar bZ   = interpolator(selectComponent(1), {phi, z, r});
		Scalar bR   = interpolator(selectComponent(2), {phi, z, r});
		
		return { bPhi, bZ, bR };
	}
	
	EIGEN_DEVICE_FUNC Vec<Scalar, 3> operator()(const TensorMap<Tensor<Scalar, 4>>& fieldData, const Vec<Scalar, 3>& xyz) {
		return operator()(TensorMap<const Tensor<Scalar, 4>>(fieldData.data(), fieldData.dimensions()), xyz);
	}
	
	EIGEN_DEVICE_FUNC Vec<Scalar, 3> inSlabOrientation(const TensorMap<Tensor<Scalar, 4>>& fieldData, const Vec<Scalar, 3>& xyz) {
		return inSlabOrientation(TensorMap<const Tensor<Scalar, 4>>(fieldData.data(), fieldData.dimensions()), xyz);
	}
};

template<typename Scalar>
struct C1CubicDeriv {
	// Cubic interpolation based on Hermite polynomials
	// Performs polynomial interpolation on [0, 1] with constraints on f(0), f'(0), f(1), and f'(1)
	//
	// Implementation notes:
	// The Hermite polynomial is given as P(x) = SUM_i:0 -> n f[x0, ..., xi] PROD_j:0 -> i-1 (x - xj)
	//
	// We have (x0, x1, x2, x3) = (0, 0, 1, 1)
	//
	// This gives us:
	//
	// f[x0] = f(0)
	// f[x0, x1] = f'(0)
	//
	// f[x0, x1, x2] = f[x1, x2] - f[x0, x1]
	//               = f[x2] - f[x1] - f[x0, x1]
    //               = f(1) - f(0) - f'(0)
	//
	// f[x0, x1, x2, x3] = f[x1, x2, x3] - f[x0, x1, x2]
	//                   = f[x2, x3] - f[x1, x2] - f[x0, x1, x2]
	//                   = f[x2, x3] - (f[x2] - f[x1]) - f[x0, x1, x2]
	//                   = f'(1) - f(1) + f(0) - f[x0, x1, x2]
	
	// For the different polynomials we get:
	// - Associated with f(0):
	//   f[x0] = 1
	//   f[x0, x1] = 0
	//   f[x0, ..., x2] = -1
	//   f[x0, ..., x3] = 2
	//    ==> P[x] = 1 - x**2 + 2 * x**2 * (x - 1) [Check]
	// - Associated with f(1):
	//   f[x0] = 0
	//   f[x0, x1] = 0
	//   f[x0, ..., x2] = 1
	//   f[x0, ..., x3] = -2
	//    ==> P[x] = x**2 - 2 * x**2 * (x - 1) [Check]
	// - Associated with f'(0):
	//   f[x0] = 0
	//   f[x0, x1] = 1
	//   f[x0, ..., x2] = -1
	//   f[x0, ..., x3] = 1
	//    ==> P[x] = x - x**2 + x**2 * (x - 1) [Check]
	// - Associated with f'(1):
	//   f[x0] = 0
	//   f[x0, x1] = 0
	//   f[x0, ..., x2] = 0
	//   f[x0, ..., x3] = 1
	//    ==> P[x] = x**2 * (x - 1) [Check]
	
	
	const Scalar f0;
	const Scalar df0;
	const Scalar df1;
	const Scalar f1_minus_f0;
	const Scalar f_x0_x1_x2;
	
	constexpr C1CubicDeriv(Scalar f0, Scalar df0, Scalar f1, Scalar df1) :
		f0(f0), df0(df0), df1(df1), f1_minus_f0(f1 - f0), f_x0_x1_x2(f1 - f0 - df0)
	{}
	
	Scalar operator()(Scalar x) {
		return
		  f0
		+ df0 * x
		+ f_x0_x1_x2 * x * x
		+ (df1 - f1_minus_f0 - f_x0_x1_x2) * x * x * (x - 1);
	}
	
	Scalar d(Scalar x) {
		return
		  df0
		+ f_x0_x1_x2 * 2 * x
		+ (df1 - f1_minus_f0 - f_x0_x1_x2) * (2 * x * (x - 1) + x * x);
	}
	
	Scalar dd(Scalar x) {
		return
		  f_x0_x1_x2 * 2
		+ (df1 - f1_minus_f0 - f_x0_x1_x2) * (2 * x + 2 * (x - 1) + 2 * x);
	}
};

}
