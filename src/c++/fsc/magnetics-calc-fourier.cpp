#include "magnetics-internal.h"

#include "nudft.h"

#include <iostream>

namespace fsc { namespace internal {

Promise<void> FieldCalculatorImpl::evalFourierSurface(EvalFourierSurfaceContext ctx) {
	auto params = ctx.getParams();
	
	auto surfaces = params.getSurfaces();
	auto surfNMax = surfaces.getNTor();
	auto surfMMax = surfaces.getMPol();
	
	auto surfNumN = 2 * surfNMax + 1;
	auto surfNumM = surfMMax + 1;
	
	uint32_t surfSym = surfaces.getToroidalSymmetry();
	uint32_t surfNTurns = surfaces.getNTurns();
	
	// Compute toroidal mode numbers & evaluation angles
	auto modeN = kj::heapArray<double>(surfNumN);
	for(auto iN : kj::indices(modeN)) {
		double baseN = iN;
		if(iN > surfNMax)
			baseN = ((int) iN) - (int) surfNumN;
		
		baseN *= surfSym;
		baseN /= surfNTurns;
		modeN[iN] = baseN;
	}
	
	auto phiVals = params.getPhi();
	auto thetaVals = params.getTheta();
	
	const size_t nPhi = phiVals.size();
	const size_t nTheta = thetaVals.size();
	
	// Read symmetric surface basis
	
	Eigen::Tensor<double, 3> rCos;
	Eigen::Tensor<double, 3> zSin;
	readVardimTensor(surfaces.getRCos(), 0, rCos);
	readVardimTensor(surfaces.getZSin(), 0, zSin);
	
	KJ_REQUIRE(rCos.dimension(0) == surfNumM);
	KJ_REQUIRE(rCos.dimension(1) == surfNumN);
	const int64_t nSurfs = rCos.dimension(2);
	
	KJ_REQUIRE(zSin.dimension(0) == surfNumM);
	KJ_REQUIRE(zSin.dimension(1) == surfNumN);
	KJ_REQUIRE(zSin.dimension(2) == nSurfs);
	
	Eigen::Tensor<double, 3> rSin;
	Eigen::Tensor<double, 3> zCos;
	
	// Read non-symmetric surface basis
	if(surfaces.isNonSymmetric()) {
		readVardimTensor(surfaces.getNonSymmetric().getZCos(), 0, zCos);
		readVardimTensor(surfaces.getNonSymmetric().getRSin(), 0, rSin);
	
		KJ_REQUIRE(zCos.dimension(0) == surfNumM);
		KJ_REQUIRE(zCos.dimension(1) == surfNumN);
		KJ_REQUIRE(zCos.dimension(2) == nSurfs);
		
		KJ_REQUIRE(rSin.dimension(0) == surfNumM);
		KJ_REQUIRE(rSin.dimension(1) == surfNumN);
		KJ_REQUIRE(rSin.dimension(2) == nSurfs);
	} else {
		rSin.resize(surfNumM, surfNumN, nSurfs);
		zCos.resize(surfNumM, surfNumN, nSurfs);
		
		rSin.setZero();
		zCos.setZero();
	}
	
	// Calculate points and derivatives on all surfaces
	
	using ADS = Eigen::AutoDiffScalar<Vec2d>;
	Eigen::Tensor<ADS, 4> points(nTheta, nPhi, nSurfs, 3);
	points.setZero();
	
	#pragma omp parallel for
	for(long int multiIndex = 0; multiIndex < nPhi * nTheta * nSurfs; ++multiIndex) {
		unsigned long tmp = multiIndex;
		
		const unsigned int iPhi = tmp % nPhi;
		tmp /= nPhi;
		
		const unsigned int iTheta = tmp % nTheta;
		tmp /= nTheta;
		
		const unsigned int iSurf = tmp;
		
		ADS phi(phiVals[iPhi], 2, 0);
		ADS theta(thetaVals[iTheta], 2, 1);
		
		for(auto m : kj::range(0, surfNumM)) {
		for(auto iN : kj::range(0, surfNumN)) {
			double n = modeN[iN];
			
			ADS cosVal = cos(-n * phi + m * theta);
			ADS sinVal = sin(-n * phi + m * theta);
			
			ADS rContrib = rCos(m, iN, iSurf) * cosVal + rSin(m, iN, iSurf) * sinVal;
			ADS zContrib = zCos(m, iN, iSurf) * cosVal + zSin(m, iN, iSurf) * sinVal;
			
			ADS xContrib = cos(phi) * rContrib;
			ADS yContrib = sin(phi) * rContrib;
			
			points(iTheta, iPhi, iSurf, 0) += xContrib;
			points(iTheta, iPhi, iSurf, 1) += yContrib;
			points(iTheta, iPhi, iSurf, 2) += zContrib;
		}
		}
	}
	
	// Take apart autodiff scalar into elements
	
	Eigen::Tensor<double, 4> val(nTheta, nPhi, nSurfs, 3);
	Eigen::Tensor<double, 4> ddPhi(nTheta, nPhi, nSurfs, 3);
	Eigen::Tensor<double, 4> ddTheta(nTheta, nPhi, nSurfs, 3);
	for(auto i : kj::indices(val)) {
		val.data()[i] = points.data()[i].value();
		ddPhi.data()[i] = points.data()[i].derivatives()[0];
		ddTheta.data()[i] = points.data()[i].derivatives()[1];
	}
	
	auto adjustShape = [&](Float64Tensor::Builder b) {
		auto is = surfaces.getRCos().getShape();
		auto os = b.initShape(is.size() + 1);
		
		os.set(0, 3);
		for(auto i : kj::range(1, os.size() - 2))
			os.set(i, is[i - 1]);
		os.set(os.size() - 2, nPhi);
		os.set(os.size() - 1, nTheta);
	};
	
	auto writeAdjusted = [&](Eigen::Tensor<double, 4>& in, Float64Tensor::Builder out) {
		writeTensor(in, out);
		adjustShape(out);
	};
	
	auto results = ctx.initResults();
	writeAdjusted(val, results.initPoints());
	writeAdjusted(ddPhi, results.initPhiDerivatives());
	writeAdjusted(ddTheta, results.initThetaDerivatives());
	
	return READY_NOW;
}

Promise<void> FieldCalculatorImpl::evalFourierSurfaceNoGrid(EvalFourierSurfaceNoGridContext ctx) {
	auto params = ctx.getParams();
	
	auto surfaces = params.getSurfaces();
	auto surfNMax = surfaces.getNTor();
	auto surfMMax = surfaces.getMPol();
	
	auto surfNumN = 2 * surfNMax + 1;
	auto surfNumM = surfMMax + 1;
	
	uint32_t surfSym = surfaces.getToroidalSymmetry();
	uint32_t surfNTurns = surfaces.getNTurns();
	
	// Compute toroidal mode numbers & evaluation angles
	auto modeN = kj::heapArray<double>(surfNumN);
	for(auto iN : kj::indices(modeN)) {
		double baseN = iN;
		if(iN > surfNMax)
			baseN = ((int) iN) - (int) surfNumN;
		
		baseN *= surfSym;
		baseN /= surfNTurns;
		modeN[iN] = baseN;
	}
	
	Eigen::Tensor<double, 1> phiVals;
	Eigen::Tensor<double, 1> thetaVals;
	
	auto vardimShape = readVardimTensor(params.getPhi(), 0, phiVals);
	readVardimTensor(params.getTheta(), 0, thetaVals);
	
	KJ_REQUIRE(phiVals.dimension(0) == thetaVals.dimension(0), "Phi and Theta tensors must have identical no. of elements");
	
	const size_t nPoints = phiVals.dimension(0);
	
	// Read symmetric surface basis
	
	Eigen::Tensor<double, 3> rCos;
	Eigen::Tensor<double, 3> zSin;
	readVardimTensor(surfaces.getRCos(), 0, rCos);
	readVardimTensor(surfaces.getZSin(), 0, zSin);
	
	KJ_REQUIRE(rCos.dimension(0) == surfNumM);
	KJ_REQUIRE(rCos.dimension(1) == surfNumN);
	const int64_t nSurfs = rCos.dimension(2);
	
	KJ_REQUIRE(zSin.dimension(0) == surfNumM);
	KJ_REQUIRE(zSin.dimension(1) == surfNumN);
	KJ_REQUIRE(zSin.dimension(2) == nSurfs);
	
	Eigen::Tensor<double, 3> rSin;
	Eigen::Tensor<double, 3> zCos;
	
	// Read non-symmetric surface basis
	if(surfaces.isNonSymmetric()) {
		readVardimTensor(surfaces.getNonSymmetric().getZCos(), 0, zCos);
		readVardimTensor(surfaces.getNonSymmetric().getRSin(), 0, rSin);
	
		KJ_REQUIRE(zCos.dimension(0) == surfNumM);
		KJ_REQUIRE(zCos.dimension(1) == surfNumN);
		KJ_REQUIRE(zCos.dimension(2) == nSurfs);
		
		KJ_REQUIRE(rSin.dimension(0) == surfNumM);
		KJ_REQUIRE(rSin.dimension(1) == surfNumN);
		KJ_REQUIRE(rSin.dimension(2) == nSurfs);
	} else {
		rSin.resize(surfNumM, surfNumN, nSurfs);
		zCos.resize(surfNumM, surfNumN, nSurfs);
		
		rSin.setZero();
		zCos.setZero();
	}
	
	// Calculate points and derivatives on all surfaces
	
	using ADS = Eigen::AutoDiffScalar<Vec2d>;
	Eigen::Tensor<ADS, 3> points(nPoints, nSurfs, 3);
	points.setZero();
	
	#pragma omp parallel for
	for(long int multiIndex = 0; multiIndex < nPoints * nSurfs; ++multiIndex) {
		unsigned long tmp = multiIndex;
		
		const unsigned int iPoint = tmp % nPoints;
		tmp /= nPoints;
		
		const unsigned int iSurf = tmp;
		
		ADS phi(phiVals[iPoint], 2, 0);
		ADS theta(thetaVals[iPoint], 2, 1);
		
		for(auto m : kj::range(0, surfNumM)) {
		for(auto iN : kj::range(0, surfNumN)) {
			double n = modeN[iN];
			
			ADS cosVal = cos(-n * phi + m * theta);
			ADS sinVal = sin(-n * phi + m * theta);
			
			ADS rContrib = rCos(m, iN, iSurf) * cosVal + rSin(m, iN, iSurf) * sinVal;
			ADS zContrib = zCos(m, iN, iSurf) * cosVal + zSin(m, iN, iSurf) * sinVal;
			
			ADS xContrib = cos(phi) * rContrib;
			ADS yContrib = sin(phi) * rContrib;
			
			points(iPoint, iSurf, 0) += xContrib;
			points(iPoint, iSurf, 1) += yContrib;
			points(iPoint, iSurf, 2) += zContrib;
		}
		}
	}
	
	// Take apart autodiff scalar into elements
	
	Eigen::Tensor<double, 3> val(nPoints, nSurfs, 3);
	Eigen::Tensor<double, 3> ddPhi(nPoints, nSurfs, 3);
	Eigen::Tensor<double, 3> ddTheta(nPoints, nSurfs, 3);
	for(auto i : kj::indices(val)) {
		val.data()[i] = points.data()[i].value();
		ddPhi.data()[i] = points.data()[i].derivatives()[0];
		ddTheta.data()[i] = points.data()[i].derivatives()[1];
	}
	
	auto adjustShape = [&](Float64Tensor::Builder b) {
		auto is = surfaces.getRCos().getShape();
		auto os = b.initShape(is.size() - 2 + vardimShape.size() + 1);
		
		os.set(0, 3);
		for(auto i : kj::range(0, is.size() - 2))
			os.set(i + 1, is[i]);
		for(auto i : kj::range(0, vardimShape.size()))
			os.set(i + 1 + is.size() - 2, vardimShape[i]);
	};
	
	auto writeAdjusted = [&](Eigen::Tensor<double, 3>& in, Float64Tensor::Builder out) {
		writeTensor(in, out);
		adjustShape(out);
	};
	
	auto results = ctx.initResults();
	writeAdjusted(val, results.initPoints());
	writeAdjusted(ddPhi, results.initPhiDerivatives());
	writeAdjusted(ddTheta, results.initThetaDerivatives());
	
	return READY_NOW;
}

Promise<void> FieldCalculatorImpl::surfaceToFourier(SurfaceToFourierContext ctx) {
	auto surfaces = ctx.getParams().getSurfaces();
	
	auto surfNMax = surfaces.getNTor();
	auto surfMMax = surfaces.getMPol();
	
	auto surfNumN = 2 * surfNMax + 1;
	auto surfNumM = surfMMax + 1;
	
	
	Eigen::Tensor<double, 3> rCos;
	Eigen::Tensor<double, 3> zSin;
	
	readVardimTensor(surfaces.getRCos(), 0, rCos);
	readVardimTensor(surfaces.getZSin(), 0, zSin);
	
	int64_t nSurf = rCos.dimension(2);
	
	auto transformTensor = [&](const Eigen::Tensor<double, 3>& in, bool isCos) {
		KJ_REQUIRE(in.dimension(0) == surfNumM);
		KJ_REQUIRE(in.dimension(1) == surfNumN);
		KJ_REQUIRE(in.dimension(2) == nSurf);
		
		Eigen::Tensor<double, 3> out(2 * surfMMax + 1, 2 * surfNMax + 1, nSurf);
		
		for(auto iSurf : kj::range(0, nSurf)) {
			for(auto iN : kj::range(0, 2 * surfNMax + 1)) {
				auto iN2 = (iN == 0) ? 0 : 2 * surfNMax + 1 - iN;
				// Skip m == 0, n < 0 modes
				for(auto iM : kj::range(0, surfMMax + 1)) {
					auto iM2 = (iM == 0) ? 0 : 2 * surfMMax + 1 - iM;
					
					// The n < 0, m = 0 modes are empty
					if(iN > surfNMax && iM == 0) continue;
					
					// Because we use ang = m * theta - n * phi, we need
					// to use iN2 and iM together (and vice-versa)
					
					// In sin, the first one needs negative sign (a naive look
					// at Euler's formula suggest the second one, but we have a
					// 1/i multiplier in front which results in a -i for the
					// complex parts, which changes the sign.
					
					out(iM, iN2, iSurf) = isCos ? in(iM, iN, iSurf) / 2 : -in(iM, iN, iSurf) / 2;
					out(iM2, iN, iSurf) = in(iM, iN, iSurf) / 2;
				}
			}
			
			out(0, 0, iSurf) = in(0, 0, iSurf);
		}
		
		return out;
	};
			
	auto adjustShape = [&](Float64Tensor::Builder out) {
		auto surfShape = surfaces.getRCos().getShape();
		auto shape = out.initShape(surfShape.size());
		for(auto i : kj::range(0, surfShape.size() - 2))
			shape.set(i, surfShape[i]);
		shape.set(shape.size() - 2, 2 * surfNMax + 1);
		shape.set(shape.size() - 1, 2 * surfMMax + 1);
	};
	
	auto transform = [&](const Eigen::Tensor<double, 3>& in, bool symmetric, Float64Tensor::Builder out) {
		writeTensor(transformTensor(in, symmetric), out);
		adjustShape(out);
	};
	
	auto zero = [&](Float64Tensor::Builder out) {
		Tensor<double, 3> tmp(2 * surfMMax + 1, 2 * surfNMax + 1, nSurf);
		tmp.setZero();
		writeTensor(tmp, out);
		adjustShape(out);
	};
	
	transform(rCos, true, ctx.getResults().getRReal());
	transform(zSin, false, ctx.getResults().getZImag());
	
	if(surfaces.isNonSymmetric()) {
		Tensor<double, 3> zCos;
		Tensor<double, 3> rSin;
		
		readVardimTensor(surfaces.getNonSymmetric().getZCos(), 0, zCos);
		readVardimTensor(surfaces.getNonSymmetric().getRSin(), 0, rSin);
		
		transform(zCos, true, ctx.getResults().getZReal());
		transform(rSin, false, ctx.getResults().getRImag());
	} else {
		zero(ctx.getResults().getZReal());
		zero(ctx.getResults().getRImag());
	}
	
	return READY_NOW;
}

Promise<void> FieldCalculatorImpl::calculateRadialModes(CalculateRadialModesContext ctx) {
	auto params = ctx.getParams();
	
	return kj::startFiber(65536 * 8, [this, ctx](kj::WaitScope& ws) mutable {			
		auto params = ctx.getParams();
		auto surfaces = params.getSurfaces();
		
		auto gcd = [&](unsigned int a, unsigned int b) {
			while(true) {
				if(b == 0) return a;
				a %= b;
				std::swap(b, a);
			}
		};
		
		const size_t nSym = params.getNSym();
		const double phiMultiplier = static_cast<double>(surfaces.getNTurns()) / nSym;
		
		auto phiVals = kj::heapArray<double>(params.getNPhi());
		for(auto iPhi : kj::indices(phiVals)) {
			phiVals[iPhi] = 2 * fsc::pi * phiMultiplier / phiVals.size() * iPhi;
		}
		
		auto thetaVals = kj::heapArray<double>(params.getNTheta());
		for(auto iTheta : kj::indices(thetaVals)) {
			thetaVals[iTheta] = 2 * fsc::pi / thetaVals.size() * iTheta;
		}
		
		const int nPhi = params.getNPhi();
		const int nTheta = params.getNTheta();
		
		Tensor<double, 4> surfPoints;
		Tensor<double, 4> surfPhiDeriv;
		Tensor<double, 4> surfThetaDeriv;
		
		// Calculate evaluation points for surfaces
		{
			auto req = thisCap().evalFourierSurfaceRequest();
			req.setSurfaces(surfaces);
			req.setPhi(phiVals);
			req.setTheta(thetaVals);
			
			auto resp = req.send().wait(ws);
			readVardimTensor(resp.getPoints(), 1, surfPoints);
			readVardimTensor(resp.getPhiDerivatives(), 1, surfPhiDeriv);
			readVardimTensor(resp.getThetaDerivatives(), 1, surfThetaDeriv);
			
			const int nSurfs = surfPoints.dimension(2);
			
			KJ_REQUIRE(surfPoints.dimension(0) == nTheta);
			KJ_REQUIRE(surfPoints.dimension(1) == nPhi);
			KJ_REQUIRE(surfPoints.dimension(2) == nSurfs);
			KJ_REQUIRE(surfPoints.dimension(3) == 3);
			
			KJ_REQUIRE(surfPhiDeriv.dimension(0) == nTheta);
			KJ_REQUIRE(surfPhiDeriv.dimension(1) == nPhi);
			KJ_REQUIRE(surfPhiDeriv.dimension(2) == nSurfs);
			KJ_REQUIRE(surfPhiDeriv.dimension(3) == 3);
			
			KJ_REQUIRE(surfThetaDeriv.dimension(0) == nTheta);
			KJ_REQUIRE(surfThetaDeriv.dimension(1) == nPhi);
			KJ_REQUIRE(surfThetaDeriv.dimension(2) == nSurfs);
			KJ_REQUIRE(surfThetaDeriv.dimension(3) == 3);
		}
		
		const int nSurfs = surfPoints.dimension(2);
		
		// Calculate radial basis at evaluation points
		Tensor<double, 4> radialBasis(nTheta, nPhi, nSurfs, 3);
		
		for(int iTheta : kj::range(0, nTheta)) {
		for(int iPhi : kj::range(0, nPhi)) {
		for(int iSurf : kj::range(0, nSurfs)) {
			Vec3d ePhi(
				surfPhiDeriv(iTheta, iPhi, iSurf, 0),
				surfPhiDeriv(iTheta, iPhi, iSurf, 1),
				surfPhiDeriv(iTheta, iPhi, iSurf, 2)
			);
			Vec3d eTheta(
				surfThetaDeriv(iTheta, iPhi, iSurf, 0),
				surfThetaDeriv(iTheta, iPhi, iSurf, 1),
				surfThetaDeriv(iTheta, iPhi, iSurf, 2)
			);
			Vec3d eRad = ePhi.cross(eTheta);
			
			if(params.getQuantity() == FieldCalculator::RadialModeQuantity::FIELD)
				eRad /= eRad.norm();
			
			radialBasis(iTheta, iPhi, iSurf, 0) = eRad(0);
			radialBasis(iTheta, iPhi, iSurf, 1) = eRad(1);
			radialBasis(iTheta, iPhi, iSurf, 2) = eRad(2);
		}}}
		
		// Calculate field values at evaluation points
		Tensor<double, 3> radialValues;
		{
			auto req = thisCap().evaluateXyzRequest();
			req.setField(params.getField());
			writeTensor(surfPoints, req.initPoints());
			
			auto resp = req.send().wait(ws);
			
			Tensor<double, 4> fieldValues;
			readTensor(resp.getValues(), fieldValues);
			
			radialValues = (fieldValues * radialBasis).sum(Eigen::array<int, 1>({3}));
		}
		
		// Normalize against background field
		if(params.hasBackground()) {
			auto req = thisCap().evaluateXyzRequest();
			req.setField(params.getBackground());
			writeTensor(surfPoints, req.initPoints());
			
			auto resp = req.send().wait(ws);
			
			Tensor<double, 4> fieldValues;
			readTensor(resp.getValues(), fieldValues);
		
			for(int iTheta : kj::range(0, nTheta)) {
			for(int iPhi : kj::range(0, nPhi)) {
			for(int iSurf : kj::range(0, nSurfs)) {
				Vec3d backgroundField(
					fieldValues(iTheta, iPhi, iSurf, 0),
					fieldValues(iTheta, iPhi, iSurf, 1),
					fieldValues(iTheta, iPhi, iSurf, 2)
				);
				double norm = backgroundField.norm();
				radialValues(iTheta, iPhi, iSurf) /= norm;
			}}}
		}
		
		// If we have VMEC fourier convention (ang = m * theta - n * phi),
		// swap the order along the phi coordinate
		if(params.getFourierConvention() == FieldCalculator::FourierConvention::VMEC) {
			auto tmp = radialValues;
			
			for(int64_t i : kj::range(0, nPhi)) {
				int64_t i2 = i == 0 ? 0 : nPhi - i;
				radialValues.chip(i, 1) = tmp.chip(i2, 1);
			}
		}
		
		// Prepare modes to calculate
		const int nMax = params.getNMax();
		const int mMax = params.getMMax();
		
		const int numN = 2 * nMax + 1;
		const int numM = mMax + 1;
		
		// Run NUDFT (Non-uniform DFT)
		Tensor<double, 3> cosCoeffs(numM, numN, nSurfs);
		Tensor<double, 3> sinCoeffs(numM, numN, nSurfs);
		
		Tensor<double, 3> dftReal(2 * mMax + 1, 2 * nMax + 1, nSurfs);
		Tensor<double, 3> dftImag(2 * mMax + 1, 2 * nMax + 1, nSurfs);
		
		sinCoeffs.setZero();
		cosCoeffs.setZero();
		
		dftReal.setZero();
		dftImag.setZero();
		
		bool canUseFft = params.getNPhi() == 2 * params.getNMax() + 1 && params.getNTheta() == 2 * params.getMMax() + 1;
		
		if(params.getUseFFT() && canUseFft) {
			// Fast path using 2D FFT
			KJ_DBG("FFT fast path enabled");
			
			// Checked before
			KJ_ASSERT(nPhi == 2 * nMax + 1);
			KJ_ASSERT(nTheta == 2 * mMax + 1);
			
			std::array<int, 2> dims = {0, 1};
			Tensor<std::complex<double>, 3> fft = radialValues.fft<Eigen::BothParts, Eigen::FFT_FORWARD>(dims);
			
			double scale = 0;
			switch(params.getFourierNormalization()) {
				case FieldCalculator::FourierNormalization::UNNORMALIZED:
					scale = 1;
					break;
				case FieldCalculator::FourierNormalization::L2_PRESERVING:
					scale = 1.0 / sqrt(nPhi * nTheta);
					break;
				case FieldCalculator::FourierNormalization::NORMALIZED:
					scale = 1.0 / (nPhi * nTheta);
					break;
				default:
					KJ_FAIL_REQUIRE("Unknown normalization spec");
			}
			
			dftReal = fft.real() * scale;
			dftImag = fft.imag() * scale;
			
			// Compute cos and sin coefficients
			for(auto iSurf : kj::range(0, nSurfs)) {
				// THESE ONES HAVE NO NEGATIVE FREQ COUNTERPART
				// SO NO FACTOR 2 !!!
				cosCoeffs(0, 0, iSurf) = dftReal(0, 0, iSurf);
				sinCoeffs(0, 0, iSurf) = dftImag(0, 0, iSurf);
			
				for(auto in : kj::range(1, nMax + 1)) {
					cosCoeffs(0, in, iSurf) = 2 * dftReal(0, in, iSurf);
					sinCoeffs(0, in, iSurf) = 2 * dftImag(0, in, iSurf);
				}
				
				for(auto im : kj::range(1, mMax + 1)) {
					cosCoeffs(im, 0, iSurf) = 2 * dftReal(im, 0, iSurf);
					sinCoeffs(im, 0, iSurf) = 2 * dftImag(im, 0, iSurf);
				}
				
				for(auto in : kj::range(1, numN)) {
					for(auto im : kj::range(1, numM)) {
						auto in2 = 2 * nMax + 1 - in;
						auto im2 = 2 * mMax + 1 - im;
						
						cosCoeffs(im, in, iSurf) = 2 * dftReal(im, in, iSurf); //+ dftReal(im2, in2, Eigen::all);
						sinCoeffs(im, in, iSurf) = 2 * dftImag(im, in, iSurf); //- dftImag(im2, in2, Eigen::all);
					}
				}
			}
			
			sinCoeffs = -sinCoeffs; // e(ix) - e(-ix) is 2i * sin. ai e(ix) - ai(e-ix) is -2asin
			
		} else {
			KJ_DBG("Using slow NUDFT path");
			using FP = nudft::FourierPoint<2, 1>;
			using FM = nudft::FourierMode<2, 1>;
						
			kj::Vector<FM> modes;
			modes.reserve(mMax * (2 * nMax + 1));
			
			for(int iM : kj::range(0, numM)) {
			for(int iN : kj::range(0, numN)) {
				int m = iM;
				int n = iN <= nMax ? iN : iN - numN;
				
				if(m == 0 && n < 0)
					continue;
				
				FM mode;
				mode.coeffs[0] = n;
				mode.coeffs[1] = m;
				modes.add(mode);
			}}
			
			for(int iSurf : kj::range(0, nSurfs)) {				
				kj::Vector<FP> points;
				points.reserve(nPhi * nTheta);
				
				for(int iPhi : kj::range(0, nPhi)) {
				for(int iTheta : kj::range(0, nTheta)) {
					FP newPoint;
					
					newPoint.angles[0] = phiVals[iPhi] / phiMultiplier;
					newPoint.angles[1] = thetaVals[iTheta];
					newPoint.y[0] = radialValues(iTheta, iPhi, iSurf);
					
					points.add(newPoint);
				}}
				
				nudft::calculateModes<2, 1>(points, modes);
				
				for(auto& mode : modes) {
					int in = mode.coeffs[0];
					int im = mode.coeffs[1];
					if(in < 0) in += numN;
					
					cosCoeffs(im, in, iSurf) = mode.cosCoeffs[0];
					sinCoeffs(im, in, iSurf) = mode.sinCoeffs[0];
				}
			}
			
			double scale = 0;
			switch(params.getFourierNormalization()) {
				case FieldCalculator::FourierNormalization::UNNORMALIZED:
					scale = nPhi * nTheta;
					break;
				case FieldCalculator::FourierNormalization::L2_PRESERVING:
					scale = sqrt(nPhi * nTheta);
					break;
				case FieldCalculator::FourierNormalization::NORMALIZED:
					scale = 1.0;
					break;
				default:
					KJ_FAIL_REQUIRE("Unknown normalization spec");
			}
			
			cosCoeffs = cosCoeffs * scale;
			sinCoeffs = sinCoeffs * scale;
			
			// Compute Re and Im coefficients
			for(auto iSurf : kj::range(0, nSurfs)) {
				dftReal(0, 0, iSurf) = cosCoeffs(0, 0, iSurf);
				dftImag(0, 0, iSurf) = sinCoeffs(0, 0, iSurf);
			
				for(auto in : kj::range(1, nMax + 1)) {
					auto in2 = 2 * nMax + 1 - in;
					
					dftReal(0, in, iSurf) = 0.5 * cosCoeffs(0, in, iSurf);
					dftImag(0, in, iSurf) = 0.5 * sinCoeffs(0, in, iSurf);
					
					dftReal(0, in2, iSurf) =  0.5 * cosCoeffs(0, in, iSurf);
					dftImag(0, in2, iSurf) = -0.5 * sinCoeffs(0, in, iSurf);
				}
				
				for(auto im : kj::range(1, mMax + 1)) {
					auto im2 = 2 * mMax + 1 - im;
					
					dftReal(im, 0, iSurf) = 0.5 * cosCoeffs(im, 0, iSurf);
					dftImag(im, 0, iSurf) = 0.5 * sinCoeffs(im, 0, iSurf);
					
					dftReal(im2, 0, iSurf) =  0.5 * cosCoeffs(im, 0, iSurf);
					dftImag(im2, 0, iSurf) = -0.5 * sinCoeffs(im, 0, iSurf);
				}
				
				for(auto in : kj::range(1, numN)) {
					for(auto im : kj::range(1, numM)) {
						auto in2 = 2 * nMax + 1 - in;
						auto im2 = 2 * mMax + 1 - im;
						
						dftReal(im, in, iSurf) = 0.5 * cosCoeffs(im, in, iSurf);
						dftImag(im, in, iSurf) = 0.5 * sinCoeffs(im, in, iSurf);
						
						dftReal(im2, in2, iSurf) =  0.5 * cosCoeffs(im, in, iSurf);
						dftImag(im2, in2, iSurf) = -0.5 * sinCoeffs(im, in, iSurf);
					}
				}
			}
			
			dftImag = -dftImag; // e(ix) - e(-ix) is 2i * sin. ai e(ix) - ai(e-ix) is -2asin
		}
		
		// Write results
		auto results = ctx.initResults();
		
		// Cos coeffs
		writeTensor(cosCoeffs, results.initCosCoeffs());
		writeTensor(sinCoeffs, results.initSinCoeffs());
		
		auto adjustShape = [&](Float64Tensor::Builder out) {
			auto surfShape = surfaces.getRCos().getShape();
			auto shape = out.initShape(surfShape.size());
			for(auto i : kj::range(0, surfShape.size() - 2))
				shape.set(i, surfShape[i]);
			shape.set(shape.size() - 2, numN);
			shape.set(shape.size() - 1, numM);
		};
		adjustShape(results.getCosCoeffs());
		adjustShape(results.getSinCoeffs());
		
		// Fourier components
		writeTensor(dftReal, results.initReCoeffs());
		writeTensor(dftImag, results.initImCoeffs());
		
		auto adjustShape2 = [&](Float64Tensor::Builder out) {
			auto surfShape = surfaces.getRCos().getShape();
			auto shape = out.initShape(surfShape.size());
			for(auto i : kj::range(0, surfShape.size() - 2))
				shape.set(i, surfShape[i]);
			shape.set(shape.size() - 2, 2 * nMax + 1);
			shape.set(shape.size() - 1, 2 * mMax + 1);
		};
		adjustShape2(results.getReCoeffs());
		adjustShape2(results.getImCoeffs());

		// Write radial values tensor
		{
			writeTensor(radialValues, results.getRadialValues());
			auto surfShape = surfaces.getRCos().getShape();
			auto shape = results.getRadialValues().initShape(surfShape.size());
			for(auto i : kj::range(0, surfShape.size() - 2))
				shape.set(i, surfShape[i]);
			shape.set(shape.size() - 2, nPhi);
			shape.set(shape.size() - 1, nTheta);
		}
		
		auto mOut = results.initMPol(numM);
		for(auto i : kj::indices(mOut))
			mOut.set(i, i);
		
		auto nOut = results.initNTor(numN);
		for(int i : kj::indices(nOut))
			nOut.set(i, (i <= nMax ? i : i - numN) / phiMultiplier);
		
		results.setPhi(phiVals);
		results.setTheta(thetaVals);
	});
}

}}
