#include "common.h"
#include "eigen.h"
#include "tensor.h"
#include "data.h"

#include "flt-kernels.h" // for kmath::wrap

#include <fsc/flt.capnp.h>
#include <fsc/services.capnp.h>
#include <fsc/index.capnp.h>

namespace fsc {
	
struct MapperImpl : public Mapper::Server {
	FLT::Client flt;
	KDTreeService::Client indexer;
	
	MapperImpl(FLT::Client flt, KDTreeService::Client indexer) : flt(mv(flt)), indexer(mv(indexer)) {}
	
	Promise<void> computeMapping(ComputeMappingContext ctx) {
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
		
		auto params = ctx.getParams();
		
		// Check that startPoints has right shape
		auto spShape = params.getStartPoints().getShape();
		KJ_REQUIRE(spShape.size() >= 1);
		KJ_REQUIRE(spShape[0] == 3);
		
		size_t shapeProd = 1;
		for(auto e : spShape)
			shapeProd *= e;
		
		auto spData = params.getStartPoints().getData();
		KJ_REQUIRE(spData.size() == shapeProd);
		
		size_t nFilaments = shapeProd / 3;
		
		auto pcRequest = flt.traceRequest();
		
		// Compute starting points for filament triplets
		double dx = params.getDx();
		
		Tensor<double, 3> reqSP(nFilaments, 3, 3);
		for(auto iPoint : kj::range(0, nFilaments)) {
			Vec3d x(spData[iPoint], spData[iPoint + nFilaments], spData[iPoint + 2 * nFilaments]);
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
		
		// Set phi planes
		auto nPhi = params.getNPhi();
		auto planes = pcRequest.initPlanes(nPhi);
		for(auto i : kj::indices(planes)) {
			planes[i].getOrientation().setPhi(2 * pi / planes.size() * i);
		}
		
		// Set other properties for tracing requests
		pcRequest.setDistanceLimit(params.getFilamentLength());
		pcRequest.setField(params.getField());
		
		KJ_DBG(pcRequest.asReader());
		
		// Perform tracing
		return pcRequest.send()
		.then([ctx, params, nFilaments, nPhi, this](capnp::Response<FLTResponse> traceResponse) mutable {
			auto nTurns = traceResponse.getNTurns();
			
			KJ_DBG("Tracing successful");
			KJ_DBG(traceResponse.getPoincareHits());
			
			// Shape nTurns, nPhi, nFilaments, 3, (x, y, z, lc_bwd, lc_fwd)
			Tensor<double, 5> pcPoints;
			readTensor(traceResponse.getPoincareHits(), pcPoints);
			
			struct IndexEntry {
				uint32_t iFilament;
				uint32_t iStartPoint;
				Vec3d x;
			};
			
			kj::Vector<IndexEntry> indexEntries;
			
			Temporary<FieldlineMapping> result;
			auto filaments = result.initFilaments(nFilaments);
			
			for(uint32_t iFilament : kj::range(0, nFilaments)) {
				// Holder struct for sorting the Poincare hits by connection length
				struct Entry {
					double lBwd;
					uint32_t iTurn;
					uint32_t iPhi;
				};
				
				kj::Vector<Entry> entries;
				
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
						
						entries.add(Entry { fabs(pcPoints(iTurn, iFilament, 0, iPhi, 4)), (uint32_t) iTurn, (uint32_t) iPhi });
					}
				}
				
				if(entries.size() < 2)
					continue;
				
				std::sort(entries.begin(), entries.end(), [](Entry& e1, Entry& e2) { return e1.lBwd < e2.lBwd; });
				
				auto filament = filaments[iFilament];
				auto data = filament.initData(6 * entries.size());
				
				// Process all entries
				kj::Vector<double> phiValues;
				double phiStart = 0;
				double currentPhi = 0;
				
				for(auto iEntry : kj::indices(entries)) {
					auto& e = entries[iEntry];
					
					KJ_DBG(e.lBwd, e.iTurn, e.iPhi);
					
					auto iTurn = e.iTurn;
					auto iPhi = e.iPhi;
					
					Vec3d x1(pcPoints(iTurn, iFilament, 0, iPhi, 0), pcPoints(iTurn, iFilament, 0, iPhi, 1), pcPoints(iTurn, iFilament, 0, iPhi, 2));
					Vec3d x2(pcPoints(iTurn, iFilament, 1, iPhi, 0), pcPoints(iTurn, iFilament, 1, iPhi, 1), pcPoints(iTurn, iFilament, 1, iPhi, 2));
					Vec3d x3(pcPoints(iTurn, iFilament, 2, iPhi, 0), pcPoints(iTurn, iFilament, 2, iPhi, 1), pcPoints(iTurn, iFilament, 2, iPhi, 2));
					
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
					
					data.set(6 * iEntry + 0, r1);
					data.set(6 * iEntry + 1, z1);
					
					for(auto i : kj::range(0, 4)) {
						data.set(6 * iEntry + 2 + i, jacobian.data()[i]);
					}
					
					double phi1 = atan2(x1[1], x1[0]);
					currentPhi += kmath::wrap(phi1 - currentPhi);
					
					phiValues.add(currentPhi);
					
					indexEntries.add(IndexEntry {
						(uint32_t) iFilament, (uint32_t) iEntry, x1
					});
					
					if(iEntry == 0)
						phiStart = currentPhi;
				}
				
				filament.setPhiStart(phiStart);
				filament.setPhiEnd(currentPhi);
				filament.setNIntervals(entries.size() - 1);
				KJ_DBG(filament);
			}
			
			KJ_DBG(indexEntries.size());
			
			// Create indexing request
			size_t MAX_CHUNK_SIZE = 500000000 / 3;
			auto indexRequest = indexer.buildRequest();
			
			BalancedIntervalSplit split(indexEntries.size(), MAX_CHUNK_SIZE);
			KJ_DBG(split.blockCount());
			
			auto chunks = indexRequest.initChunks(split.blockCount());
			for(auto i : kj::range(0, split.blockCount())) {
				KJ_DBG(i, split.edge(i + 1), split.edge(i));
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
				KJ_DBG(key, e.iFilament, e.iStartPoint);
				
				for(auto i : kj::range(0, 3)) {
					chunk.getBoxes().getData().set(3 * offset + i, e.x[i]);
				}
				chunk.getKeys().set(offset, key);
			}
			
			KJ_DBG(indexRequest);
			
			return indexRequest.send()
			.then([ctx, result = mv(result)](capnp::Response<KDTree> tree) mutable {
				result.setIndex(tree);
				
				auto published = getActiveThread().dataService().publish(mv(result));
				ctx.getResults().setMapping(published);
			});
		});
	}
};

Mapper::Client newMapper(FLT::Client flt, KDTreeService::Client indexer) {
	return kj::heap<MapperImpl>(mv(flt), mv(indexer));
}

}