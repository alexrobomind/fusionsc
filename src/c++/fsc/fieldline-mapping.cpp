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
	struct DirectionTrace {			
		struct IndexEntry {
			uint32_t iFilament;
			uint32_t iStartPoint;
			Vec3d x;
		};
		
		FLT::Client flt;
		MappingRequest::Reader params;
		FieldlineMapping::Direction::Builder result;
		bool forward;
		
		capnp::List<double>::Reader startPointData;
		size_t nFilaments;
		uint32_t nSym;
				
		kj::Vector<IndexEntry> indexEntries;
		
		DirectionTrace(FLT::Client flt, MappingRequest::Reader params, bool forward, FieldlineMapping::Direction::Builder result) :
			flt(flt), params(params), result(result), forward(forward), nFilaments(0), nSym(params.getNSym())
		{
			auto spShape = params.getStartPoints().getShape();
			KJ_REQUIRE(spShape.size() >= 1);
			KJ_REQUIRE(spShape[0] == 3);
			
			size_t shapeProd = 1;
			for(auto e : spShape)
				shapeProd *= e;
			
			startPointData = params.getStartPoints().getData();
			KJ_REQUIRE(startPointData.size() == shapeProd);
			
			nFilaments = shapeProd / 3;
			
			result.initFilaments(nFilaments * nSym);
		}
		
		Promise<void> run(size_t batchSize) {
			return runStep(0, batchSize);
		}
		
		Promise<void> runStep(size_t batchOffset, size_t batchSize) {
			// Implementation notes:
			// The mapping points are computed by making Poincaré tracing requests
			// These requests do not arrive in order of the field hit, but are primarily sorted
			// by the phi plane they arrive in. Since this might not be ordered correctly, the hits
			// first need to be sorted by their backward connection length (Note: Usually we get -Lc
			// as a result since we do not hit any geometry going forward).
			//
			// The requests are made for triplets of fieldlines starting at identical phi points, but at
			// positions (r, z), (r + dx, z) and (r, z + dx). The differences are then used to compute the
			// mapping transforms in the r-z plane.
			//
			// Requests are batched in chunks to limit the memory consumption.
			
			if(batchOffset + batchSize > nFilaments)
				batchSize = nFilaments - batchOffset;
			
			if(batchSize == 0)
				return READY_NOW;
			
			KJ_DBG("FLM batch process", forward, batchOffset, batchSize, nFilaments);
			
			auto pcRequest = flt.traceRequest();
			
			// Compute starting points for filament triplets
			double dx = params.getDx();
			
			Tensor<double, 3> reqSP(batchSize, 3, 3);
			for(auto iPoint : kj::range(0, batchSize)) {
				Vec3d x(
					startPointData[iPoint + batchOffset],
					startPointData[iPoint + batchOffset + nFilaments],
					startPointData[iPoint + batchOffset + 2 * nFilaments]
				);
				double phi = atan2(x[1], x[0]);
				double r = sqrt(x[0] * x[0] + x[1] * x[1]);
				double z = x[2];
				
				Vec3d x2(cos(phi) * (r + dx), sin(phi) * (r + dx), z);
				Vec3d x3(x[0], x[1], z + dx);
				
				for(auto iDim : kj::range(0, 3)) {
					reqSP(iPoint, 0, iDim) = x[iDim];
					reqSP(iPoint, 1, iDim) = x2[iDim];
					reqSP(iPoint, 2, iDim) = x3[iDim];
				}
			}
			
			writeTensor(reqSP, pcRequest.getStartPoints());
			// KJ_DBG(pcRequest.getStartPoints().getShape());
			
			// Set phi planes
			auto nPhi = params.getNPhi();
			auto planes = pcRequest.initPlanes(nPhi);
			for(auto i : kj::indices(planes)) {
				planes[i].getOrientation().setPhi(2 * pi / planes.size() * i);
			}
			
			// Set other properties for tracing requests
			pcRequest.setDistanceLimit(params.getFilamentLength());
			pcRequest.setField(params.getField());
			pcRequest.setForward(forward);
			pcRequest.setStepSize(params.getStepSize());
			
			return pcRequest.send()
			.then([this, nPhi, batchSize, batchOffset](capnp::Response<FLTResponse> traceResponse) mutable {
				auto nTurns = traceResponse.getNTurns();
				
				KJ_DBG("Tracing successful");
				// KJ_DBG(traceResponse.getPoincareHits().getShape());
				
				// Shape nTurns, nFilaments, 3, nPHi, (x, y, z, lc_fwd, lc_bwd)
				Tensor<double, 5> pcPoints;
				readTensor(traceResponse.getPoincareHits(), pcPoints);
				
				auto filaments = result.getFilaments();
			
				for(uint32_t iFilament : kj::range(0, batchSize)) {
					// Holder struct for sorting the Poincare hits by connection length
					struct Entry {
						double lBwd;
						uint32_t iTurn;
						uint32_t iPhi;
					};
					
					kj::Vector<Entry> entries;
					
					double lBwdMin = std::numeric_limits<double>::infinity();
					double lFwdMin = std::numeric_limits<double>::infinity();
					
					// Copy out all potential hits
					for(uint32_t iPhi : kj::range(0, nPhi)) {
						for(uint32_t iTurn : kj::range(0, nTurns)) {
							// Check whether point is valid
							bool isValid = true;
							for(auto i : kj::range(0, 3)) {
								if(pcPoints(iTurn, iFilament, i, iPhi, 0) != pcPoints(iTurn, iFilament, i, iPhi, 0))
									isValid = false;
							}
							
							if(!isValid)
								continue;
							
							// KJ_DBG(iFilament, iPhi, iTurn);
							
							double lcFwd = fabs(pcPoints(iTurn, iFilament, 0, iPhi, 3));
							double lcBwd = fabs(pcPoints(iTurn, iFilament, 0, iPhi, 4));
							
							lBwdMin = std::min(lBwdMin, lcBwd);
							lFwdMin = std::min(lFwdMin, lcFwd);
							
							entries.add(Entry { fabs(pcPoints(iTurn, iFilament, 0, iPhi, 4)), (uint32_t) iTurn, (uint32_t) iPhi });
						}
					}
				
					if(entries.size() < 2)
						continue;
					
					std::sort(entries.begin(), entries.end(), [](Entry& e1, Entry& e2) { return e1.lBwd < e2.lBwd; });
					
					for(uint32_t iSym : kj::range(0, nSym)) {
						uint32_t realFilament = nSym * (iFilament + batchOffset) + iSym;
						
						auto filament = filaments[realFilament];
						auto data = filament.initData(6 * entries.size());
						
						// Process all entries
						kj::Vector<double> phiValues;
						double phiStart = 0;
						double currentPhi = 0;
						
						double angle = 2 * iSym * pi / nSym;
						Mat3d rot;
						rot.setZero();
						
						rot(0, 0) = cos(angle); rot(0, 1) = -sin(angle);
						rot(1, 0) = sin(angle); rot(1, 1) = cos(angle);
						rot(2, 2) = 1;
											
						for(auto iEntry : kj::indices(entries)) {
							auto& e = entries[iEntry];
							
							// KJ_DBG(e.lBwd, e.iTurn, e.iPhi);
							
							auto iTurn = e.iTurn;
							auto iPhi = e.iPhi;
							
							Vec3d x1(pcPoints(iTurn, iFilament, 0, iPhi, 0), pcPoints(iTurn, iFilament, 0, iPhi, 1), pcPoints(iTurn, iFilament, 0, iPhi, 2));
							Vec3d x2(pcPoints(iTurn, iFilament, 1, iPhi, 0), pcPoints(iTurn, iFilament, 1, iPhi, 1), pcPoints(iTurn, iFilament, 1, iPhi, 2));
							Vec3d x3(pcPoints(iTurn, iFilament, 2, iPhi, 0), pcPoints(iTurn, iFilament, 2, iPhi, 1), pcPoints(iTurn, iFilament, 2, iPhi, 2));
							
							//KJ_DBG((x2 - x1).norm());
							//KJ_DBG((x3 - x1).norm());
							
							double r1 = sqrt(x1[0] * x1[0] + x1[1] * x1[1]);
							double r2 = sqrt(x2[0] * x2[0] + x2[1] * x2[1]);
							double r3 = sqrt(x3[0] * x3[0] + x3[1] * x3[1]);
							
							double z1 = x1[2];
							double z2 = x2[2];
							double z3 = x3[2];
							
							Mat2d jacobian;
							jacobian(0, 0) = r2 - r1;
							jacobian(0, 1) = r3 - r1;
							jacobian(1, 0) = z2 - z1;
							jacobian(1, 1) = z3 - z1;
							
							data.set(6 * iEntry + 0, r1 + 0.5 * (r2 - r1) + 0.5 * (r3 - r1));
							data.set(6 * iEntry + 1, z1 + 0.5 * (z2 - z1) + 0.5 * (z3 - z1));
							
							for(auto i : kj::range(0, 4)) {
								data.set(6 * iEntry + 2 + i, jacobian.data()[i]);
							}
							
							double phi1 = atan2(x1[1], x1[0]) + 2 * iSym * pi / nSym;
							currentPhi += kmath::wrap(phi1 - currentPhi);
							
							phiValues.add(currentPhi);
								
							double lcFwd = fabs(pcPoints(iTurn, iFilament, 0, iPhi, 3));
							double lcBwd = fabs(pcPoints(iTurn, iFilament, 0, iPhi, 4));
							
							bool addToIndex = true;
							if(lcFwd < lFwdMin + params.getCutoff())
								addToIndex = false;
							
							if(lcBwd < lBwdMin + params.getCutoff())
								addToIndex = false;
							
							if(addToIndex) {
								indexEntries.add(IndexEntry {
									realFilament, (uint32_t) iEntry, rot * (x1 + 0.5 * (x2 - x1) + 0.5 * (x3 - x1))
								});
							}
							
							if(iEntry == 0)
								phiStart = currentPhi;
						}
						
						filament.setPhiStart(phiStart);
						filament.setPhiEnd(currentPhi);
						filament.setNIntervals(entries.size() - 1);
						// KJ_DBG(filament);
					}
				}
			})
			.then([this, batchOffset, batchSize]() {
				return runStep(batchOffset + batchSize, batchSize);
			});
		}
		
		Promise<void> buildIndex(KDTreeService::Client indexer) {
			KJ_DBG(indexEntries.size());
			
			// Create indexing request
			size_t MAX_CHUNK_SIZE = 500000000 / 3;
			auto indexRequest = indexer.buildRequest();
			
			BalancedIntervalSplit split(indexEntries.size(), MAX_CHUNK_SIZE);
			// KJ_DBG(split.blockCount());
			
			auto chunks = indexRequest.initChunks(split.blockCount());
			for(auto i : kj::range(0, split.blockCount())) {
				// KJ_DBG(i, split.edge(i + 1), split.edge(i));
				auto chunkSize = split.edge(i + 1) - split.edge(i);
				
				auto chunk = chunks[i];
				auto shape = chunk.getBoxes().initShape(2);
				shape.set(0, chunkSize);
				shape.set(1, 3);
				
				chunk.getBoxes().initData(3 * chunkSize);
				
				chunk.initKeys(chunkSize);
			}
			
			for(auto iEntry : kj::indices(indexEntries)) {
				auto iChunk = split.interval(iEntry);
				auto offset = iEntry - split.edge(iChunk);
				
				auto chunk = chunks[iChunk];
				
				auto& e = indexEntries[iEntry];
				
				uint64_t key = e.iFilament;
				key = key << 32;
				key |= e.iStartPoint;
				// KJ_DBG(key, e.iFilament, e.iStartPoint);
				
				for(auto i : kj::range(0, 3)) {
					chunk.getBoxes().getData().set(3 * offset + i, e.x[i]);
				}
				chunk.getKeys().set(offset, key);
			}
			
			// KJ_DBG(indexRequest);
			
			return indexRequest.send()
			.then([this](capnp::Response<KDTree> tree) mutable {
				result.setIndex(tree);
			});
		}
	};
	
	struct RFLMSectionTrace {
		double phi1;
		double phi2;
		
		size_t numTracingPlanes;
		size_t numPaddingPlanes;
		
		capnp::List<double>::Reader rVals;
		capnp::List<double>::Reader zVals;
		
		FLT::Client flt;
		
		RFLMRequest::Reader input;
		ReversibleFieldLineMapping::Section::Builder output;
		
		RFLMSectionTrace(FLT::Client nFlt, RFLMRequest::Reader input, ReversibleFieldLineMapping::Section::Builder output) : flt(mv(nFlt)), input(mv(input)), output(mv(output)) {
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
			
			// Note that our planes can go past the range of the section. This is because
			// we have padding planes on both sides used to extend the interpolation region.
			auto planes = req.initPlanes(numTracingPlanes + numPaddingPlanes);
			for(auto iPlane : kj::indices(planes)) {
				auto plane = planes[iPlane];
				plane.getOrientation().setPhi(phiStart + (iPlane + 1) * dPhi);
			}
			KJ_DBG(ccw, req);
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
	
	Promise<void> computeDirection(MappingRequest::Reader params, bool forward, FieldlineMapping::Direction::Builder result) {
		auto trace = heapHeld<DirectionTrace>(flt, params, forward, result);
		
		return trace -> run(params.getBatchSize())
		.then([this, trace, result]() mutable {
			return trace -> buildIndex(indexer);
			// KJ_DBG(result);
		}).attach(trace.x());
	}
	
	Promise<void> computeMapping(ComputeMappingContext ctx) {		
		auto params = ctx.getParams();
		
		KJ_REQUIRE(params.hasField(), "Must specify magnetic field");
		
		Temporary<FieldlineMapping> result;

		auto promise = computeDirection(params, true, result.getFwd())
		.then([this, bwd = result.getBwd(), params]() mutable {
			return computeDirection(params, false, bwd);
		});
		
		promise = promise.then([this, ctx, result = mv(result)]() mutable {
			auto published = getActiveThread().dataService().publish(mv(result));
			ctx.getResults().setMapping(published);
		});
		
		return promise;
	}
	
	Promise<void> computeRFLM(ComputeRFLMContext ctx) {
		auto params = ctx.getParams();
		auto planes = params.getMappingPlanes();
		
		KJ_REQUIRE(params.hasField(), "Must specify magnetic field");
		
		Temporary<ReversibleFieldLineMapping> result;
		result.setSurfaces(planes);
		result.setNPad(params.getNumPaddingPlanes());
		
		auto promiseBuilder = kj::heapArrayBuilder<Promise<void>>(planes.size());
		
		auto sections = result.initSections(planes.size());
		
		double totalWidth = 0;
		for(size_t i1 : kj::indices(planes)) {
			size_t i2 = (i1 + 1) % planes.size();
			
			auto trace = heapHeld<RFLMSectionTrace>(flt, params, sections[i1]);
			
			trace -> phi1 = planes[i1];
			trace -> phi2 = planes[i2];
			trace -> numTracingPlanes = params.getNumPlanes();
			trace -> numPaddingPlanes = params.getNumPaddingPlanes();
			trace -> rVals = params.getGridR();
			trace -> zVals = params.getGridZ();
			
			totalWidth += trace -> computeWidth();
			
			promiseBuilder.add(trace -> run().attach(trace.x()));
		}
		
		KJ_REQUIRE(std::abs(totalWidth - 2 * pi) < pi, "Sections must be of non-zero width, and angles must be specified in counter-clockwise direction");
				
		auto joined = kj::joinPromises(promiseBuilder.finish());
		return joined.then([ctx = mv(ctx), result = mv(result)]() mutable {
			ctx.initResults().setMapping(getActiveThread().dataService().publish(mv(result)));
		});
	}
};

Mapper::Client newMapper(FLT::Client flt, KDTreeService::Client indexer) {
	return kj::heap<MapperImpl>(mv(flt), mv(indexer));
}

}