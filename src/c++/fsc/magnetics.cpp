#include <capnp/message.h>

#include "magnetics.h"
#include "data.h"
#include "interpolation.h"
#include "nudft.h"

#include "kernels/launch.h"
#include "kernels/tensor.h"
#include "kernels/message.h"
#include "kernels/karg.h"

#include "magnetics-kernels.h"

namespace fsc {
	
namespace {
	
struct FieldCalculation {
	using FieldValues = ::fsc::kernels::FieldValues;
	using Field = ::fsc::kernels::Field;
	
	using MFilament = ::fsc::kernels::MFilament;
	
	constexpr static unsigned int GRID_VERSION = 7;
	
	Own<DeviceBase> _device;
	
	DeviceMappingType<FieldValues> field;
	DeviceMappingType<FieldValues> points;
	
	// This promise makes sure we only schedule a
	// calculation once the previous is finished
	Promise<void> calculation = READY_NOW;
	
	FieldCalculation(FieldValues&& pointsIn, DeviceBase& device) :
		_device(device.addRef()),
		field(mapToDevice(FieldValues(pointsIn.dimension(0), 3), device, true)),
		points(mapToDevice(mv(pointsIn), device, true))
	{
		field -> getHost().setZero();
		
		field -> updateDevice();
		points -> updateDevice();
	}
	
	~FieldCalculation() {}
	
	void addComputed(double scale, Float64Tensor::Reader otherFieldIn, ToroidalGridStruct otherGrid) {
		auto shape = otherFieldIn.getShape();
		
		KJ_REQUIRE(shape.size() == 4);
		KJ_REQUIRE(shape[3] == 3);
		
		auto data = otherFieldIn.getData();
		
		// Write field into native format
		// for(auto i : otherFieldIn.getShape())
		// 	KJ_DBG("Shape", i);
	
		// KJ_DBG(3, otherGrid.nR, otherGrid.nZ, otherGrid.nPhi);
		Field otherField(3, otherGrid.nR, otherGrid.nZ, otherGrid.nPhi);
		KJ_REQUIRE(otherField.size() == data.size());
		
		for(int i = 0; i < otherField.size(); ++i) {
			otherField.data()[i] = data[i];
		}
				
		calculation = calculation.then([this, otherField = mv(otherField), otherGrid, scale]() mutable {
			return FSC_LAUNCH_KERNEL(
				kernels::addFieldInterpKernel,
				*_device, 
				field -> getHost().dimension(0),
				FSC_KARG(field, NOCOPY), FSC_KARG(points, NOCOPY),
				FSC_KARG(otherField, ALIAS_IN), otherGrid,
				scale
			);
		});
		
	}
	
	void biotSavart(double current, Float64Tensor::Reader input, BiotSavartSettings::Reader settings) {
		auto shape = input.getShape();
		
		KJ_REQUIRE(shape.size() == 2);
		KJ_REQUIRE(shape[1] == 3);
		KJ_REQUIRE(shape[0] >= 2);
		
		int n_points = (int) shape[0];
		MFilament filament(3, n_points);
		
		// Copy filament into native buffer
		auto data = input.getData();
		for(int i = 0; i < n_points; ++i) {
			filament(0, i) = data[3 * i + 0];
			filament(1, i) = data[3 * i + 1];
			filament(2, i) = data[3 * i + 2];
		}
		
		double coilWidth = settings.getWidth();
		double stepSize  = settings.getStepSize();
		
		KJ_REQUIRE(stepSize != 0, "Please specify a step size in the Biot-Savart settings");
		
		calculation = calculation.then([this, filament = mv(filament), coilWidth, stepSize, current]() mutable {
			KJ_LOG(INFO, "Processing coil", current, coilWidth, stepSize, filament.dimension(1));
			
			// Launch calculation
			return FSC_LAUNCH_KERNEL(
				kernels::biotSavartKernel,
				*_device,
				field -> getHost().dimension(0), FSC_KARG(points, NOCOPY),
				FSC_KARG(filament, ALIAS_IN), current, coilWidth, stepSize, FSC_KARG(field, NOCOPY)
			);
		});
	}
	
	void dipoles(double scale, DipoleCloud::Reader cloud) {
		Tensor<double, 2> positions;
		Tensor<double, 2> moments;
		
		readTensor(cloud.getPositions(), positions);
		readTensor(cloud.getMagneticMoments(), moments);
		
		size_t nPoints = positions.dimension(0);
		KJ_REQUIRE(moments.dimension(0) == nPoints);
		KJ_REQUIRE(moments.dimension(1) == 3);
		KJ_REQUIRE(positions.dimension(1) == 3);
		
		auto radii = cloud.getRadii();
		KJ_REQUIRE(radii.size() == nPoints);
		
		auto radiiNative = kj::heapArray<double>(nPoints);
		for(auto i : kj::indices(radii))
			radiiNative[i] = radii[i];
		
		calculation = calculation.then([this, nPoints, scale, positions = mv(positions), moments = mv(moments), radiiNative = mv(radiiNative)]() mutable {
			return FSC_LAUNCH_KERNEL(
				kernels::dipoleFieldKernel,
				*_device,
				field -> getHost().dimension(0), FSC_KARG(points, NOCOPY),
				FSC_KARG(mv(positions), ALIAS_IN), FSC_KARG(mv(moments), ALIAS_IN), FSC_KARG(mv(radiiNative), ALIAS_IN),
				scale, FSC_KARG(field, NOCOPY)
			);
		});
	}
	
	void equilibrium(double scale, AxisymmetricEquilibrium::Reader equilibrium) {		
		calculation = calculation.then([this, equilibrium = Temporary<AxisymmetricEquilibrium>(equilibrium), scale]() mutable {
			auto mapped = FSC_MAP_BUILDER(fsc, AxisymmetricEquilibrium, mv(equilibrium), *_device, true);
			
			return FSC_LAUNCH_KERNEL(
				kernels::eqFieldKernel,
				*_device,
				field -> getHost().dimension(0),
				FSC_KARG(points, NOCOPY),
				FSC_KARG(mapped, ALIAS_IN), scale, FSC_KARG(field, NOCOPY)
			);
		});
	}
	
	Promise<void> finish(Float64Tensor::Builder out) {
		calculation = calculation
		.then([this]() {
			field -> updateHost();
			return _device -> barrier();
		})
		.then([this, out]() {
			writeTensor(field->getHost(), out);
		});
		
		return mv(calculation);
	}
};

template<typename Device>
struct CalculationSession : public FieldCalculator::Server {	
	// Device device;
	Own<DeviceBase> device;
	
	CalculationSession(Own<DeviceBase> device) :
		device(mv(device))
	{}
	
	Promise<void> evaluateXyz(EvaluateXyzContext context) {
		KJ_LOG(INFO, "Initiating magnetic field evaluation");
		
		auto field = context.getParams().getField();
		auto pointsIn = context.getParams().getPoints();
		
		// Validate shape of input tensor
		validateTensor(pointsIn);
		auto pointsShape = pointsIn.getShape();
		KJ_REQUIRE(pointsIn.getShape().size() >= 1);
		KJ_REQUIRE(pointsIn.getShape()[0] == 3);
		
		size_t nPoints = 1;
		for(auto i : kj::range(1, pointsShape.size()))
			nPoints *= pointsShape[i];
		
		auto pointData = pointsIn.getData();
		
		Eigen::Tensor<double, 2> points(nPoints, 3);
		for(auto i : kj::range(0, nPoints)) {
			points(i, 0) = pointData[0 * nPoints + i];
			points(i, 1) = pointData[1 * nPoints + i];
			points(i, 2) = pointData[2 * nPoints + i];
		}
		
		return processRoot(field, mv(points), context.getResults().initValues())
		.then([context, pointsShape]() mutable {
			// Adjust field shape to match points shape
			context.getResults().getValues().setShape(pointsShape);
		});
	}
	
	Promise<void> evaluatePhizr(EvaluatePhizrContext context) {
		auto xyzReq = thisCap().evaluateXyzRequest();
		auto params = context.getParams();
		
		xyzReq.setField(params.getField());
		xyzReq.setPoints(params.getPoints());
		
		auto points = xyzReq.getPoints().getData();
		auto nPoints = points.size() / 3;
		
		for(auto i : kj::range(0, nPoints)) {
			double phi = points[0 * nPoints + i];
			double z = points[1 * nPoints + i];
			double r = points[2 * nPoints + i];
			
			double x = std::cos(phi) * r;
			double y = std::sin(phi) * r;
			
			points.set(0 * nPoints + i, x);
			points.set(1 * nPoints + i, y);
			points.set(2 * nPoints + i, z);
		}
		
		return xyzReq.send().then([params, nPoints, context](auto response) mutable {
			auto results = context.initResults();
			results.setValues(response.getValues());
			
			auto points = params.getPoints().getData();
			auto values = results.getValues().getData();
			
			KJ_REQUIRE(values.size() == 3 * nPoints);
			
			for(auto i : kj::range(0, nPoints)) {
				double bX = values[0 * nPoints + i];
				double bY = values[1 * nPoints + i];
				double bZ = values[2 * nPoints + i];
				
				double phi = points[0 * nPoints + i];
				
				double bR = bX * std::cos(phi) + bY * std::sin(phi);
				double bPhi = bY * std::cos(phi) - bX * std::sin(phi);
				
				values.set(0 * nPoints + i, bPhi);
				values.set(1 * nPoints + i, bZ);
				values.set(2 * nPoints + i, bR);
			}
		});
	}
	
	//! Handles compute request
	Promise<void> compute(ComputeContext context) {
		KJ_LOG(INFO, "Initiating magnetic field computation");
		
		constexpr unsigned int GRID_VERSION = 7;
		auto grid = readGrid(context.getParams().getGrid(), GRID_VERSION);
		
		// Calculate all grid points
		Tensor<double, 4> points(grid.nPhi, grid.nZ, grid.nR, 3);
		for(auto iR : kj::range(0, grid.nR)) {
			for(auto iPhi : kj::range(0, grid.nPhi)) {
				for(auto iZ : kj::range(0, grid.nZ)) {
					points(iPhi, iZ, iR, 0) = grid.phi(iPhi);
					points(iPhi, iZ, iR, 1) = grid.z(iZ);
					points(iPhi, iZ, iR, 2) = grid.r(iR);
				}
			}
		}
		
		// Submit evaluation request request
		auto calcReq = thisCap().evaluatePhizrRequest();
		calcReq.setField(context.getParams().getField());
		writeTensor(points, calcReq.getPoints());
		
		// Parameters no longer required
		context.releaseParams();
		
		auto refPromise = calcReq.send().then([grid](auto response) -> DataRef<Float64Tensor>::Client {
			Tensor<double, 4> values;
			readTensor(response.getValues(), values);
			
			Temporary<Float64Tensor> holder;
			
			// The computation request uses the format (phi, z, r, 3) (column major)
			// We need to transpose to (3, r, z, phi)
			{
				using A = Eigen::array<Eigen::Index, 4>;
				Tensor<double, 4> tmp = values.shuffle<A>({3, 2, 1, 0});
				values = tmp;
			}
			
			writeTensor(values, holder.asBuilder());
			
			return getActiveThread().dataService().publish(holder.asReader());
		});
			
		auto cf = context.initResults().getComputedField();
		writeGrid(grid, cf.initGrid());
		cf.setData(mv(refPromise));
		
		return READY_NOW;
	}
	
	Promise<void> interpolateXyz(InterpolateXyzContext ctx) {
		auto params = ctx.getParams();
		
		auto req = thisCap().evaluateXyzRequest();
		
		req.getField().setComputedField(params.getField());
		req.setPoints(params.getPoints());
		
		return ctx.tailCall(mv(req));
	}
	
	Promise<void> evalFourierSurface(EvalFourierSurfaceContext ctx) {
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
	
	Promise<void> calculateRadialModes(CalculateRadialModesContext ctx) {
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
			
			// Prepare modes to calculate
			const int nMax = params.getNMax();
			const int mMax = params.getMMax();
			
			const int numN = 2 * nMax + 1;
			const int numM = mMax + 1;
			
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
			
			// Run NUDFT (Non-uniform DFT)
			Tensor<double, 3> cosCoeffs(numM, numN, nSurfs);
			Tensor<double, 3> sinCoeffs(numM, numN, nSurfs);
			
			sinCoeffs.setZero();
			cosCoeffs.setZero();
			
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
			
			// Write results
			auto results = ctx.initResults();
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
		
	//! Processes a root node of a magnetic field (creates calculator)
	Promise<void> processRoot(MagneticField::Reader node, Eigen::Tensor<double, 2>&& points, Float64Tensor::Builder out) {		
		auto newCalculator = heapHeld<FieldCalculation>(mv(points), *device);
		
		auto calcDone = processField(*newCalculator, node, 1);
		
		return calcDone.then([newCalculator, out, this]() mutable {				
			return newCalculator -> finish(out).eagerlyEvaluate(nullptr);
		})
		.attach(newCalculator.x());
	}
	
	Promise<void> processFilament(FieldCalculation& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, double scale) {
		if(scale == 0)
			return READY_NOW;
		
		while(node.isNested()) {
			node = node.getNested();
		}
		
		switch(node.which()) {
			case Filament::INLINE:
				// The biot savart operation is chained by the calculator
				calculator.biotSavart(scale, node.getInline(), settings);
				return READY_NOW;				
				
			case Filament::REF:
				return getActiveThread().dataService().download(node.getRef()).then([&calculator, settings, scale, this](LocalDataRef<Filament> local) mutable {
					return processFilament(calculator, local.get(), settings, scale).attach(cp(local));
				});
			
			case Filament::SUM: {
				auto sum = node.getSum();
				auto arrBuilder = kj::heapArrayBuilder<Promise<void>>(sum.size());
				
				for(auto i : kj::indices(sum)) {
					arrBuilder.add(processFilament(calculator, sum[i], settings, scale));
				}
				
				return kj::joinPromises(arrBuilder.finish());
			}
			default:
				KJ_FAIL_REQUIRE("Unresolved filament node encountered during magnetic field calculation", node);
		}
	}
	
	Promise<void> processField(FieldCalculation& calculator, MagneticField::Reader node, double scale) {	
		if(scale == 0)
			return READY_NOW;
		
		while(node.isNested()) {
			node = node.getNested();
		}
		
		switch(node.which()) {
			case MagneticField::SUM: {
				auto builder = kj::heapArrayBuilder<Promise<void>>(node.getSum().size());
				
				for(auto newNode : node.getSum()) {
					builder.add(processField(calculator, newNode, scale));
				}
				
				return kj::joinPromises(builder.finish());
			}
			case MagneticField::REF: {
				// First download the ref
				auto ref = node.getRef();
				return getActiveThread().dataService().download(ref)
				.then([&calculator, scale, this](LocalDataRef<MagneticField> local) {
					// Then process it like usual
					return processField(calculator, local.get(), scale).attach(cp(local));
				});
			}
			case MagneticField::COMPUTED_FIELD: {
				auto cField = node.getComputedField();
				
				constexpr unsigned int GRID_VERSION = 7;
				ToroidalGridStruct grid = readGrid(cField.getGrid(), GRID_VERSION);
				
				// Then download data				
				return getActiveThread().dataService().download(cField.getData())
				.then([&calculator, scale, grid](LocalDataRef<Float64Tensor> field) {					
					calculator.addComputed(scale, field.get(), grid);
				});
			}
			case MagneticField::FILAMENT_FIELD: {
				// Process Biot-Savart field
				auto fField = node.getFilamentField();
				
				return processFilament(calculator, fField.getFilament(), fField.getBiotSavartSettings(), scale * fField.getCurrent() * fField.getWindingNo());
			}
			case MagneticField::SCALE_BY: {
				auto scaleBy = node.getScaleBy();
				
				return processField(calculator, scaleBy.getField(), scale * scaleBy.getFactor());
			}
			case MagneticField::INVERT: {
				return processField(calculator, node.getInvert(), -scale);
			}
			case MagneticField::NESTED: {
				return processField(calculator, node.getNested(), scale);
			}
			case MagneticField::CACHED: {
				auto cached = node.getCached();
				
				// Check if all points fit
				auto hostPoints = calculator.points -> getHost();
				auto nPoints = hostPoints.dimension(0);
				
				ToroidalGridStruct grid;
				try {
					constexpr unsigned int GRID_VERSION = 7;
					grid = readGrid(cached.getComputed().getGrid(), GRID_VERSION);
				} catch(kj::Exception&) {
					goto recalculate;
				}
				
				for(auto iPoint : kj::range(0, nPoints)) {
					double x = hostPoints(iPoint, 0);
					double y = hostPoints(iPoint, 1);
					double z = hostPoints(iPoint, 2);
					
					double r = sqrt(x*x + y*y);
					if(r < grid.rMin || r > grid.rMax)
						goto recalculate;
					if(z < grid.zMin || z > grid.zMax)
						goto recalculate;
				}
							
				return getActiveThread().dataService().download(cached.getComputed().getData())
				.then([&calculator, scale, grid](LocalDataRef<Float64Tensor> field) {					
					calculator.addComputed(scale, field.get(), grid);
				});
				
				// Jump to this label when we can not use the cached field
				recalculate:
				return processField(calculator, cached.getNested(), scale);
			}
			case MagneticField::AXISYMMETRIC_EQUILIBRIUM: {
				calculator.equilibrium(scale, node.getAxisymmetricEquilibrium());
				return READY_NOW;
			}
			case MagneticField::DIPOLE_CLOUD: {
				calculator.dipoles(scale, node.getDipoleCloud());
				return READY_NOW;
			}
			default:
				KJ_FAIL_REQUIRE("Unresolved magnetic field node encountered during field calculation.", node);
		}
	//});
	}
};

struct FieldCacheResolver : public FieldResolverBase {
	ID target;
	Temporary<ComputedField> compField;
	
	FieldCacheResolver(ID target, Temporary<ComputedField> compField) :
		target(mv(target)),
		compField(mv(compField))
	{}
		
	Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) override {
		return ID::fromReaderWithRefs(input)
		.then([this, input, output, context](ID id) mutable -> Promise<void> {
			if(id == target) {
				auto cached = output.initCached();
				cached.setComputed(compField);
				cached.setNested(input);
				
				return READY_NOW;
			}
		
			return FieldResolverBase::processField(input, output, context);
		});
	}
};

}

bool isBuiltin(MagneticField::Reader field) {
	switch(field.which()) {
		case MagneticField::SUM:
		case MagneticField::REF:
		case MagneticField::COMPUTED_FIELD:
		case MagneticField::FILAMENT_FIELD:
		case MagneticField::SCALE_BY:
		case MagneticField::INVERT:
		case MagneticField::CACHED:
		case MagneticField::NESTED:
			return true;
		default:
			return false;
	}
}

bool isBuiltin(Filament::Reader filament) {
	switch(filament.which()) {
		case Filament::INLINE:
		case Filament::REF:
		case Filament::NESTED:
		case Filament::SUM:
			return true;
		
		default:
			return false;
	}
}

FieldResolver::Client newCache(MagneticField::Reader field, ComputedField::Reader computed) {
	Temporary<ComputedField> cf(computed);
	auto clientPromise = ID::fromReaderWithRefs(field).then([cf = mv(cf)](ID id) mutable -> FieldResolver::Client {
		return kj::heap<FieldCacheResolver>(mv(id), mv(cf));
	});
	
	return clientPromise;
}
	
ToroidalGridStruct readGrid(ToroidalGrid::Reader in, unsigned int maxOrdinal) {
	KJ_REQUIRE(hasMaximumOrdinal(in, maxOrdinal));
	
	ToroidalGridStruct out;
	
	out.rMin = in.getRMin();
	out.rMax = in.getRMax();
	out.zMin = in.getZMin();
	out.zMax = in.getZMax();
	out.nR = (int) in.getNR();
	out.nZ = (int) in.getNZ();
	out.nSym = (int) in.getNSym();
	out.nPhi = (int) in.getNPhi();
	
	KJ_REQUIRE(out.isValid(), in);
	return out;
}

void writeGrid(const ToroidalGridStruct& in, ToroidalGrid::Builder out) {
	KJ_REQUIRE(in.isValid());
	
	out.setRMin(in.rMin);
	out.setRMax(in.rMax);
	out.setZMin(in.zMin);
	out.setZMax(in.zMax);
	out.setNR(in.nR);
	out.setNZ(in.nZ);
	out.setNSym(in.nSym);
	out.setNPhi(in.nPhi);
}

Promise<void> FieldResolverBase::resolveField(ResolveFieldContext context) {
	auto input = context.getParams().getField();
	auto output = context.initResults();
	
	return processField(input, output, context);
}

Promise<void> FieldResolverBase::resolveFilament(ResolveFilamentContext ctx) {
	auto input = ctx.getParams().getFilament();
	
	auto req = thisCap().resolveFieldRequest();
	req.getField().initFilamentField().setFilament(input);
	req.setFollowRefs(ctx.getParams().getFollowRefs());
	
	return req.send()
	.then([ctx](auto field) mutable {
		KJ_REQUIRE(field.isFilamentField());
		
		ctx.setResults(field.getFilamentField().getFilament());
	});
}

Promise<void> FieldResolverBase::processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) {	
	switch(input.which()) {
		case MagneticField::SUM: {
			auto inSum = input.getSum();
			auto outSum = output.initSum(inSum.size());
			
			auto subTasks = kj::heapArrayBuilder<Promise<void>>(inSum.size());
			for(unsigned int i = 0; i < inSum.size(); ++i) {
				subTasks.add(processField(inSum[i], outSum[i], context));
			}
			
			return kj::joinPromises(subTasks.finish()).attach(thisCap());
		}
		case MagneticField::REF: {
			if(!context.getParams().getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<capnp::MallocMessageBuilder>();
			MagneticField::Builder newOutput = tmpMessage -> initRoot<MagneticField>();
			
			return getActiveThread().dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<MagneticField> ref) mutable {
				return processField(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(getActiveThread().dataService().publish(newOutput));
			}).attach(mv(tmpMessage), thisCap());
		}
		case MagneticField::COMPUTED_FIELD: {
			output.setComputedField(input.getComputedField());
			return kj::READY_NOW;
		}
		case MagneticField::FILAMENT_FIELD: {
			auto filIn  = input.getFilamentField();
			auto filOut = output.initFilamentField();
			
			filOut.setCurrent(filIn.getCurrent());
			filOut.setBiotSavartSettings(filIn.getBiotSavartSettings());
			filOut.setWindingNo(filIn.getWindingNo());
			
			return processFilament(filIn.getFilament(), filOut.initFilament(), context);
		}
		case MagneticField::SCALE_BY: {
			output.initScaleBy().setFactor(input.getScaleBy().getFactor());
			return processField(input.getScaleBy().getField(), output.getScaleBy().initField(), context);
		}
		case MagneticField::INVERT: {
			return processField(input.getInvert(), output.initInvert(), context);
		}
		case MagneticField::NESTED: {
			return processField(input.getNested(), output, context);
		}
		case MagneticField::CACHED: {
			auto cachedIn = input.getCached();
			auto cachedOut = output.getCached();
			
			cachedOut.setComputed(cachedIn.getComputed());
			
			return processField(cachedIn.getNested(), cachedOut.getNested(), context);
		}
		default: {
			output.setNested(input);
			return READY_NOW;
		}
	}
}

Promise<void> FieldResolverBase::processFilament(Filament::Reader input, Filament::Builder output, ResolveFieldContext context) {
	switch(input.which()) {
		case Filament::INLINE: {
			output.setInline(input.getInline());
			return kj::READY_NOW;
		}
		case Filament::REF: {
			if(!context.getParams().getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<capnp::MallocMessageBuilder>();
			Filament::Builder newOutput = tmpMessage -> initRoot<Filament>();
			
			return getActiveThread().dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<Filament> ref) mutable {
				return processFilament(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(getActiveThread().dataService().publish(newOutput));
			}).attach(mv(tmpMessage), thisCap());
		}
		case Filament::NESTED: {
			return processFilament(input.getNested(), output, context);
		}
		case Filament::SUM: {
			auto sumIn = input.getSum();
			auto sumOut = output.initSum(sumIn.size());
			
			auto arrBuilder = kj::heapArrayBuilder<Promise<void>>(sumIn.size());
			
			for(auto i : kj::indices(sumIn)) {
				auto in = sumIn[i];
				auto out = sumOut[i];
				
				arrBuilder.add(processFilament(in, out, context));
			}
			
			return kj::joinPromises(arrBuilder.finish());
		}
		default: {
			output.setNested(input);
			return READY_NOW;
		}
	}
}

FieldCalculator::Client newFieldCalculator(Own<DeviceBase> dev) {
	return kj::heap<CalculationSession<Eigen::ThreadPoolDevice>>(mv(dev));
}

namespace {
	void buildCoil(double rMaj, double rMin, double phi, Filament::Builder out) {
		const size_t nTheta = 4;
		
		Tensor<double, 2> result(3, nTheta);
		for(auto i : kj::range(0, nTheta)) {
			double theta = 2 * pi / nTheta * i;
			
			double r = rMaj + rMin * std::cos(theta);
			double z =        rMin * std::sin(theta);
			
			double x = std::cos(phi) * r;
			double y = std::sin(phi) * r;
			
			result(0, i) = x;
			result(1, i) = y;
			result(2, i) = z;
		}
		
		writeTensor(result, out.initInline());
	}
	
	void buildAxis(double rMaj, Filament::Builder out) {
		const size_t nPhi = 200;
		
		Tensor<double, 2> result(3, nPhi);
		for(auto i : kj::range(0, nPhi)) {
			double phi = 2 * pi / nPhi * i;
			
			double x = std::cos(phi) * rMaj;
			double y = std::sin(phi) * rMaj;
			double z = 0;
			
			result(0, i) = x;
			result(1, i) = y;
			result(2, i) = z;
		}
		
		writeTensor(result, out.initInline());
	}
}

void simpleTokamak(MagneticField::Builder output, double rMajor, double rMinor, unsigned int nCoils, double Ip) {
	nCoils = 20;
	auto fields = output.initSum(nCoils + 1);
	
	for(auto i : kj::range(0, nCoils)) {
		auto filamentField = fields[i].initFilamentField();
		filamentField.getBiotSavartSettings().setStepSize(0.01);
		
		double phi = 2 * pi / nCoils * i;
		
		buildCoil(rMajor, rMinor, phi, filamentField.getFilament());	
	}
	
	{
		auto filamentField = fields[nCoils].initFilamentField();
		filamentField.setCurrent(Ip);
		filamentField.getBiotSavartSettings().setStepSize(0.01);
		
		buildAxis(rMajor, filamentField.getFilament());
	}
}

}