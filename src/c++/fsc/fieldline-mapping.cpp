#include "common.h"
#include "eigen.h"
#include "tensor.h"
#include "data.h"

#include "flt-kernels.h" // for kmath::wrap

#include <fsc/flt.capnp.h>
#include <fsc/services.capnp.h>
#include <fsc/index.capnp.h>

namespace fsc {

namespace {	
	struct RFLMSectionTrace {
		double phi1;
		double phi2;
		
		size_t numTracingPlanes;
		size_t numPaddingPlanes;
		
		capnp::List<double>::Reader rVals;
		capnp::List<double>::Reader zVals;
		
		FLT::Client flt;
		
		RFLMRequest::Reader input;
		ReversibleFieldlineMapping::Section::Builder output;
		
		RFLMSectionTrace(FLT::Client nFlt, RFLMRequest::Reader input, ReversibleFieldlineMapping::Section::Builder output) : flt(mv(nFlt)), input(mv(input)), output(mv(output)) {
		}
		
		static double halfPhi(double phi1, double phi2) {
			double dPhi = phi2 - phi1;
			dPhi = fmod(dPhi, 2 * pi);
			dPhi += 2 * pi;
			dPhi = fmod(dPhi, 2 * pi);
			
			return phi1 + dPhi / 2;
		}
		
		double computeWidth() {
			double dPhi = phi2 - phi1;
			dPhi = fmod(dPhi, 2 * pi);
			dPhi += 2 * pi;
			dPhi = fmod(dPhi, 2 * pi);
			
			if(dPhi == 0)
				return 2 * pi;
			return dPhi;
		}
		
		size_t numPlanesTot() {
			return 1 + 2 * (numTracingPlanes + numPaddingPlanes);
		}
		
		void buildStartPoints(Float64Tensor::Builder out) {
			Tensor<double, 3> data(rVals.size(), zVals.size(), 3);
			
			const double phi = phi1 + 0.5 * computeWidth();
			
			for(auto iR : kj::indices(rVals)) {
				for(auto iZ : kj::indices(zVals)) {
					data(iR, iZ, 0) = cos(phi) * rVals[iR];
					data(iR, iZ, 1) = sin(phi) * rVals[iR];
					data(iR, iZ, 2) = zVals[iZ];
				}
			}
			
			writeTensor(data, out);
		}
		
		auto traceDirection(bool ccw) {
			double width = computeWidth();
			double phiStart = phi1 + 0.5 * width;
			double span = ccw ? 0.5 * width : -0.5 * width;			
			double dPhi = span / numTracingPlanes;
			
			auto hook = capnp::ClientHook::from(cp(flt));
			
			auto req = flt.traceRequest();
			req.setForward(ccw);
			req.getForwardDirection().setCcw();
			req.setField(input.getField());
			req.setTurnLimit(1);
			req.setDistanceLimit(input.getDistanceLimit());
			req.setStepSize(input.getStepSize());
			
			// Note that our planes can go past the range of the section. This is because
			// we have padding planes on both sides used to extend the interpolation region.
			auto planes = req.initPlanes(numTracingPlanes + numPaddingPlanes);
			for(auto iPlane : kj::indices(planes)) {
				auto plane = planes[iPlane];
				plane.getOrientation().setPhi(phiStart + (iPlane + 1) * dPhi);
			}
			buildStartPoints(req.getStartPoints());
			
			return req.send().dropPipeline();
		}
		
		void processTraces(Float64Tensor::Reader fwdTensorR, Float64Tensor::Reader bwdTensorR) {
			int64_t nPhiTot = numPlanesTot();
			Tensor<double, 3> rOut(rVals.size(), zVals.size(), nPhiTot);
			Tensor<double, 3> zOut(rVals.size(), zVals.size(), nPhiTot);
			Tensor<double, 3> lenOut(rVals.size(), zVals.size(), nPhiTot);
			
			int64_t halfPoint = numTracingPlanes + numPaddingPlanes;
			
			// KJ_DBG(fwdTensorR);
			
			auto pFwdTensor = fsc::mapTensor<Tensor<double, 5>>(fwdTensorR);
			auto pBwdTensor = fsc::mapTensor<Tensor<double, 5>>(bwdTensorR);
			
			auto& fwdTensor = *pFwdTensor;
			auto& bwdTensor = *pBwdTensor;
			
			for(int64_t iPhi : kj::range(0, numTracingPlanes + numPaddingPlanes)) {
				for(int64_t iR : kj::indices(rVals)) {
					for(int64_t iZ : kj::indices(zVals)) {
						double xFwd = fwdTensor(0, iR, iZ, iPhi, 0);
						double yFwd = fwdTensor(0, iR, iZ, iPhi, 1);
						double zFwd = fwdTensor(0, iR, iZ, iPhi, 2);
						double lFwd = std::abs(fwdTensor(0, iR, iZ, iPhi, 4));
						
						double xBwd = bwdTensor(0, iR, iZ, iPhi, 0);
						double yBwd = bwdTensor(0, iR, iZ, iPhi, 1);
						double zBwd = bwdTensor(0, iR, iZ, iPhi, 2);
						double lBwd = std::abs(bwdTensor(0, iR, iZ, iPhi, 4));
						
						double rFwd = sqrt(xFwd * xFwd + yFwd * yFwd);
						double rBwd = sqrt(xBwd * xBwd + yBwd * yBwd);
						
						rOut(iR, iZ, halfPoint - iPhi - 1) = rBwd;
						zOut(iR, iZ, halfPoint - iPhi - 1) = zBwd;
						lenOut(iR, iZ, halfPoint - iPhi - 1) = -lBwd;
						
						rOut(iR, iZ, halfPoint + iPhi + 1) = rFwd;
						zOut(iR, iZ, halfPoint + iPhi + 1) = zFwd;
						lenOut(iR, iZ, halfPoint + iPhi + 1) = lFwd;
					}
				}
			}
			
			for(auto iR : kj::indices(rVals)) {
				for(auto iZ : kj::indices(zVals)) {
					rOut(iR, iZ, halfPoint) = rVals[iR];
					zOut(iR, iZ, halfPoint) = zVals[iZ];
					lenOut(iR, iZ, halfPoint) = 0;
				}
			}
			
			writeTensor(rOut, output.getR());
			writeTensor(zOut, output.getZ());
			writeTensor(lenOut, output.getTraceLen());
		}
		
		Promise<void> run() {
			return traceDirection(false)
			.then([this](auto cwResponse) {
				return traceDirection(true)
				.then([this, cwResponse = mv(cwResponse)](auto ccwResponse) {
					processTraces(ccwResponse.getPoincareHits(), cwResponse.getPoincareHits());
				});
			});
		}
	};
}
	
struct MapperImpl : public Mapper::Server {
	FLT::Client flt;
	KDTreeService::Client indexer;
	
	MapperImpl(FLT::Client flt, KDTreeService::Client indexer) : flt(mv(flt)), indexer(mv(indexer)) {}
	
	Promise<void> computeRFLM(ComputeRFLMContext ctx) {
		auto params = ctx.getParams();
		auto planes = params.getMappingPlanes();
		
		KJ_REQUIRE(params.hasField(), "Must specify magnetic field");
		
		Temporary<ReversibleFieldlineMapping> result;
		result.setSurfaces(planes);
		result.setNPad(params.getNumPaddingPlanes());
		
		auto promiseBuilder = kj::heapArrayBuilder<Promise<void>>(planes.size());
		// Promise<void> computationPromise = READY_NOW;
		
		auto sections = result.initSections(planes.size());
		
		auto u0 = params.getU0();
		auto v0 = params.getV0();
		
		KJ_REQUIRE(hasMaximumOrdinal(params, 9), "You are trying to use features that this server version does not support");
		
		KJ_REQUIRE(u0.size() == 1 || u0.size() == planes.size(), "Size of u0 must be 1 or no. of planes", u0.size(), planes.size());
		KJ_REQUIRE(v0.size() == 1 || v0.size() == planes.size(), "Size of v0 must be 1 or no. of planes", v0.size(), planes.size());
		KJ_REQUIRE(params.getGridR().size() >= 2, "Must specify at least 2 R values");
		KJ_REQUIRE(params.getGridZ().size() >= 2, "Must specify at least 2 Z values");
		
		double totalWidth = 0;
		for(size_t i1 : kj::indices(planes)) {
			auto section = sections[i1];
			section.setU0(u0.size() == 1 ? u0[0] : u0[i1]);
			section.setV0(v0.size() == 1 ? v0[0] : v0[i1]);
			
			size_t i2 = (i1 + 1) % planes.size();
			
			auto trace = heapHeld<RFLMSectionTrace>(flt, params, section);
			
			trace -> phi1 = planes[i1];
			trace -> phi2 = planes[i2];
			trace -> numTracingPlanes = params.getNumPlanes();
			trace -> numPaddingPlanes = params.getNumPaddingPlanes();
			trace -> rVals = params.getGridR();
			trace -> zVals = params.getGridZ();
			
			totalWidth += trace -> computeWidth();
			
			/*computationPromise = computationPromise.then([trace]() mutable {
				return trace -> run();
			}).attach(trace.x());*/
			promiseBuilder.add(trace -> run().attach(trace.x()));
		}
		
		KJ_REQUIRE(std::abs(totalWidth - 2 * pi) < pi, "Sections must be of non-zero width, and angles must be specified in counter-clockwise direction");
				
		auto computationPromise = kj::joinPromises(promiseBuilder.finish());
		return computationPromise.then([ctx = mv(ctx), result = mv(result)]() mutable {
			ctx.initResults().setMapping(getActiveThread().dataService().publish(mv(result)));
		});
	}
};

Mapper::Client newMapper(FLT::Client flt, KDTreeService::Client indexer) {
	return kj::heap<MapperImpl>(mv(flt), mv(indexer));
}

}