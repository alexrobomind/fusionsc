#include "magnetics-internal.h"

namespace fsc { namespace internal {

namespace {

double calculateFluxOnMesh(
	Mesh::Reader mesh,
	

}

}}

namespace fsc { namespace internal {

Promise<double> FieldCalculatorImpl::calculateFluxOnTriMesh(Mesh::Reader mesh, MagneticField::Reader field) {
	// Calculate field on all mesh points
	Tensor<double, 2> points;
	readTensor(mesh.getVertices());
	
	auto computeReq = thisCap().evaluateXyzRequest();
	writeTensor(points.transpose(), computeReq.getPoints());
	computeReq.setField(field);
	
	return computeReq.send().then([mesh, points = kj::mv(points)](auto response) {
		Tensor<double, 2> values;
		readTensor(response.getValues(), values);
		
		double result = 0;
		auto processTri = [&](size_t i1, size_t i2, size_t i3) {
			#define VEC_VAL(name, var, idx) Vec3d name(var(0, idx), var(1, idx), var(2, idx))
			
			VEC_VAL(p1, points, i1);
			VEC_VAL(p2, points, i2);
			VEC_VAL(p3, points, i3);
			
			VEC_VAL(v1, values, i1);
			VEC_VAL(v2, values, i2);
			VEC_VAL(v3, values, i3);
			
			#undef VEC_VAL
			
			Vec3d orientedArea = (p2 - p1).cross(p1 - p3);
			Vec3d avgVal = (v1 + v2 + v3) / 3;
			
			result += avgVal * orientedArea;
		};
		
		auto indices = mesh.getIndices();
		if(mesh.isTriMesh()) {
			
			for(size_t i = 0; i < indices.size(); i += 3) {
				processTri(i, i + 1, i + 2);
			}
		}
	});
}

Promise<void> FieldCalculatorImpl::calculateTotalFlux(CalculateTotalFluxContext ctx) {
	auto params = ctx.getParams();
	auto surfaces = params.getSurfaces();
	
	uint32_t nTheta = params.getNTheta();
	
	// Calculate surface contour
	auto pReq = thisCap().evalFourierSurfaceRequest();
	{
		pReq.setSurfaces(surfaces);
		pReq.initPhi(1).set(0, params.getPhi());
		
		auto thetaVals = pReq.initTheta(nTheta);
		for(auto i : kj::range(0, nTheta)) {
			thetaVals.set(i, 2 * fsc::pi / nTheta * i);
		}
	}
	
	return pReq.send().then([ctx, params, nTheta, this](auto pointsResponse) mutable {
		Tensor<double, 3> points;
		Tensor<double, 3> thetaDerivs;
		
		// since nPhi is 1, we can collapse it into the vardim shape
		readVardimTensor(pointsResponse.getPoints(), 1, points);
		readVardimTensor(pointsResponse.getThetaDerivatives(), 1, thetaDerivs);
		size_t nSurf = points.dimension(1);
		
		// points array is of (fortran order) shape [nTheta, nSurf, 3]
		// Calculate mean along iTheta to get center points for integration
		std::array<int, 1> dims = {0};
		Tensor<double, 2> centers = points.mean(dims);
		
		// Calculate vector from center to outer contour
		Tensor<double, 3> dr = points;
		for(auto i : kj::range(0, nTheta)) {
			dr.chip(i, 0) -= centers;
		}
		
		// Calculate points to evaluate field on
		size_t nR = params.getNR();
		auto lambdas = kj::heapArray<double>(nR);
		for(auto i : kj::indices(lambdas)) {
			lambdas[i] = (i + 1) * 1.0 / nR;
		}
		
		Tensor<double, 4> pointsCalc(nSurf, nR, nTheta, 3);
		for(auto iSurf : kj::range(0, nSurf)) { for(auto iR : kj::range(0, nR)) { for(uint32_t iTheta : kj::range(0, nTheta)) {
			for(int iDim : kj::range(0, 3)) {
				pointsCalc(iSurf, iR, iTheta, iDim) = lambdas[iR] * points(iTheta, iSurf, iDim) + (1 - lambdas[iR]) * centers(iSurf, iDim);
			}
		}}}
		
		// Calculate surface element ePhi * (eRmin x eTheta) across contour
		Tensor<double, 3> surfaceElements(nTheta, nSurf, 3);
		
		const double phi = params.getPhi();
		Vec3d ePhi(-sin(phi), cos(phi), 0);
		
		for(auto iTheta : kj::range(0, nTheta)) {
			for(auto iSurf : kj::range(0, nSurf)) {
				Vec3d er(
					dr(iTheta, iSurf, 0),
					dr(iTheta, iSurf, 1), 
					dr(iTheta, iSurf, 2)
				);
				Vec3d eTheta(
					thetaDerivs(iTheta, iSurf, 0),
					thetaDerivs(iTheta, iSurf, 1),
					thetaDerivs(iTheta, iSurf, 2)
				);
				
				Vec3d cross = er.cross(eTheta);
				surfaceElements(iTheta, iSurf, 0) = cross(0);
				surfaceElements(iTheta, iSurf, 1) = cross(1);
				surfaceElements(iTheta, iSurf, 2) = cross(2);
			}
		}
		
		auto req = thisCap().evaluateXyzRequest();
		writeTensor(pointsCalc, req.getPoints());
		req.setField(params.getField());
		
		return req.send().then([nSurf, nR, nTheta, params, phi, surfaceElements = mv(surfaceElements), lambdas = mv(lambdas), ctx, this](auto req) mutable {
			double dLambda = 1.0 / params.getNR();
			double dTheta = 2 * fsc::pi / nTheta;
			
			Vec3d ePhi(-sin(phi), cos(phi), 0);
			
			Tensor<double, 4> fieldVals;
			readTensor(req.getValues(), fieldVals);
			
			Tensor<double, 1> flux(nSurf);
			// Tensor<double, 1> area(nSurf);
			for(auto iSurf : kj::range(0, nSurf)) {
				double totalFlux = 0;
				// double totalArea = 0;

				for(auto iR : kj::range(0, nR)) { for(uint32_t iTheta : kj::range(0, nTheta)) {
					double contrib = 0;
					double areaContrib = 0;
					
					// Take dot product of field and surface element
					for(int iDim = 0; iDim < 3; ++iDim) {
						contrib += surfaceElements(iTheta, iSurf, iDim) * fieldVals(iSurf, iR, iTheta, iDim);
						areaContrib += surfaceElements(iTheta, iSurf, iDim) * ePhi(iDim);
					}
					
					double scale = lambdas[iR] - 0.5 * dLambda; // Use midpoint rule in minor radius integration
					scale *= dLambda * dTheta;
					
					totalFlux += contrib * scale;
					// totalArea += areaContrib * scale;
				}}
				
				flux(iSurf) = totalFlux;
				// area(iSurf) = totalArea;
			}
			
			// Use rCos tensor to extract actual vardim shape of surface array
			Tensor<double, 3> rCos;
			auto vardimShape = readVardimTensor(params.getSurfaces().getRCos(), 0, rCos);
			
			writeVardimTensor(flux, 0, vardimShape, ctx.initResults().initFlux());
			// writeVardimTensor(area, 0, vardimShape, ctx.getResults().initArea());
		});
	});
}

}}
