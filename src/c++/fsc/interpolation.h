#pragma once

#include "tensor.h"

namespace fsc {

/**
 * Simple fast linear interpolation strategy
 */
template<typename Num>
struct LinearInterpolation {
	using Scalar = Num;
	
	constexpr EIGEN_DEVICE_FUNC size_t nPoints() { return 2; }
	constexpr std::array<Scalar, 2> coefficients(Num x) {
		return {1 - x, x};
	}
	constexpr std::array<int, 2> offsets() {
		return {0, 1};
	}
};

template<int nDims, typename Strategy, int iDim>
struct NDInterpEvaluator {
	using Scalar = typename Strategy::Scalar;
	
	static_assert(iDim < nDims);
	static_assert(iDim >= 0);
	
	template<typename F, typename... Indices>
	static EIGEN_DEVICE_FUNC Scalar evaluate(Strategy& strategy, const F& f, Vec<int, nDims>& base, Vec<Scalar, nDims> lx, Indices... indices) {
		static_assert(sizeof...(indices) == iDim);
		
		Scalar result = 0;
		
		auto coefficients = strategy.coefficients(lx[iDim]);
		auto offsets      = strategy.offsets();
					
		for(int idx = 0; idx < strategy.nPoints(); ++idx) {
			result += coefficients[idx] * NDInterpEvaluator<nDims, Strategy, iDim + 1>::evaluate(strategy, f, base, lx, indices..., base[iDim] + offsets[idx]);
		}
		
		return result;
	}
};

template<int nDims, typename Strategy>
struct NDInterpEvaluator<nDims, Strategy, nDims> {
	using Scalar = typename Strategy::Scalar;
	
	template<typename F, typename... Indices>
	static EIGEN_DEVICE_FUNC Scalar evaluate(Strategy& strategy, const F& f, Vec<int, nDims>& base, Vec<Scalar, nDims> lx, Indices... indices) {
		static_assert(sizeof...(indices) == nDims);
		
		return f(indices...);
	}
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
	EIGEN_DEVICE_FUNC Scalar operator()(const F& f, Vec<Scalar, nDims> x) {
		Vec<int, nDims> base;
		Vec<Scalar, nDims> lx;
		
		for(int i = 0; i < nDims; ++i) {
			Scalar scaled = scaleMultipliers[i] * (x[i] + offsets[i]);
			base[i] = floor(scaled);
			lx[i] = scaled - base[i];
		}
		
		return NDInterpEvaluator<nDims, Strategy, 0>::evaluate(strategy, f, base, lx);
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
	
	EIGEN_DEVICE_FUNC Vec<Scalar, 3> operator()(const TensorMap<Tensor<Scalar, 4>>& fieldData, const Vec<Scalar, 3>& xyz) {
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
				if(iR > fieldData.dimension(1)) iR = fieldData.dimension(1);
				if(iZ > fieldData.dimension(2)) iZ = fieldData.dimension(2);
				
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
};

}
