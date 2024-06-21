#include <capnp/message.h>

#include "magnetics.h"
#include "magnetics-internal.h"
#include "data.h"

#include <complex>

namespace fsc { namespace internal {
	
Promise<void> FieldCalculatorImpl::evaluateXyz(EvaluateXyzContext context) {
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

Promise<void> FieldCalculatorImpl::evaluatePhizr(EvaluatePhizrContext context) {
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
Promise<void> FieldCalculatorImpl::compute(ComputeContext context) {
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

Promise<void> FieldCalculatorImpl::interpolateXyz(InterpolateXyzContext ctx) {
	auto params = ctx.getParams();
	
	auto req = thisCap().evaluateXyzRequest();
	
	req.getField().setComputedField(params.getField());
	req.setPoints(params.getPoints());
	
	return ctx.tailCall(mv(req));
}

Promise<void> FieldCalculatorImpl::surfaceToMesh(SurfaceToMeshContext ctx) {			
	auto params = ctx.getParams();
	auto surfaces = params.getSurfaces();
			
	auto phiVals = kj::heapArray<double>(params.getNPhi());
	for(auto iPhi : kj::indices(phiVals)) {
		phiVals[iPhi] = 2 * fsc::pi * surfaces.getNTurns() / phiVals.size() * iPhi;
	}
	
	auto thetaVals = kj::heapArray<double>(params.getNTheta());
	for(auto iTheta : kj::indices(thetaVals)) {
		thetaVals[iTheta] = 2 * fsc::pi / thetaVals.size() * iTheta;
	}
	
	// Calculate surface points
	auto req = thisCap().evalFourierSurfaceRequest();
	req.setSurfaces(surfaces);
	req.setPhi(phiVals); req.setTheta(thetaVals);
	
	return req.send().then([ctx, params, surfaces](auto response) mutable {
		const double shift = params.getRadialShift();
		
		// Read surface positions
		Eigen::Tensor<double, 4> pos;
		Eigen::Tensor<double, 4> dXdPhi;
		Eigen::Tensor<double, 4> dXdTheta;
		readVardimTensor(response.getPoints(), 1, pos);
		readVardimTensor(response.getPhiDerivatives(), 1, dXdPhi);
		readVardimTensor(response.getThetaDerivatives(), 1, dXdTheta);
		
		uint32_t nTheta = (uint32_t) pos.dimension(0);
		uint32_t nPhi = (uint32_t) pos.dimension(1);
		int64_t nSurf = pos.dimension(2);
		
		// auto merged = ctx.initResults().initMerged(nSurf);			
		Temporary<MergedGeometry> merged;
		auto entries = merged.initEntries(nSurf);
		
		for(auto iSurf : kj::range(0, nSurf)) {
			auto mesh = entries[iSurf].getMesh();
			
			mesh.getVertices().setShape({nTheta * nPhi, 3});
			auto meshData = mesh.getVertices().initData(3 * nTheta * nPhi);
			auto idxData = mesh.initIndices(4 * nPhi * nTheta);
			
			auto linearIndex = [&](uint32_t iPhi, uint32_t iTheta) {
				// KJ_DBG(iPhi, iTheta, nPhi, nTheta);
				iPhi %= nPhi;
				iTheta %= nTheta;
				// KJ_DBG(iPhi, iTheta);
				
				return nTheta * iPhi + iTheta;
			};
			
			for(auto iPhi : kj::range(0, nPhi)) {
				for(auto iTheta : kj::range(0, nTheta)) {
					Vec3d ePhi(
						dXdPhi(iTheta, iPhi, iSurf, 0),
						dXdPhi(iTheta, iPhi, iSurf, 1),
						dXdPhi(iTheta, iPhi, iSurf, 2)
					);
					Vec3d eTheta(
						dXdTheta(iTheta, iPhi, iSurf, 0),
						dXdTheta(iTheta, iPhi, iSurf, 1),
						dXdTheta(iTheta, iPhi, iSurf, 2)
					);
					
					Vec3d eRad = ePhi.cross(eTheta);
					eRad /= eRad.norm();
					
					Vec3d x(
						pos(iTheta, iPhi, iSurf, 0),
						pos(iTheta, iPhi, iSurf, 1),
						pos(iTheta, iPhi, iSurf, 2)
					);
					x += shift * eRad;
					
					for(size_t i = 0; i < 3; ++i)
						meshData.set(3 * nTheta * iPhi + 3 * iTheta + i, x[i]);
					
					const uint32_t offset = 4 * linearIndex(iPhi, iTheta);
					idxData.set(offset + 0, linearIndex(iPhi + 0, iTheta + 0));
					idxData.set(offset + 1, linearIndex(iPhi + 0, iTheta + 1));
					idxData.set(offset + 2, linearIndex(iPhi + 1, iTheta + 1));
					idxData.set(offset + 3, linearIndex(iPhi + 1, iTheta + 0));
				}
			}
			
			auto polyData = mesh.initPolyMesh(nPhi * nTheta + 1);
			for(auto i : kj::indices(polyData))
				polyData.set(i, 4 * i);
		}
		
		ctx.initResults().setMerged(
			getActiveThread().dataService().publish(merged.asReader())
		);
	});
}

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
			auto cachedOut = output.initCached();
			
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

Own<FieldCalculator::Server> newFieldCalculator(Own<DeviceBase> dev) {
	return kj::heap<internal::FieldCalculatorImpl>(mv(dev));
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
