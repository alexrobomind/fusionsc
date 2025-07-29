#include "common.h"
#include "eigen.h"
#include "tensor.h"
#include "data.h"

#include "flt-kernels.h" // for kmath::wrap

#include "kernels/tensor.h"
#include "kernels/karg.h"
#include "kernels/launch.h"

#include "fieldline-mapping-kernels.h"

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
		
		Own<DeviceBase> device;
		
		RFLMSectionTrace(FLT::Client nFlt, RFLMRequest::Reader input, ReversibleFieldlineMapping::Section::Builder output, DeviceBase& dev) : flt(mv(nFlt)), input(mv(input)), output(mv(output)), device(dev.addRef()) {
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
			
			auto sscIn = input.getStepSizeControl();
			auto sscOut = req.getStepSizeControl();
			
			if(sscIn.isAdaptive())
				sscOut.setAdaptive(sscIn.getAdaptive());
			
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
		
		Promise<void> processTraces(Float64Tensor::Reader fwdTensorR, Float64Tensor::Reader bwdTensorR) {
			output.setPhiStart(phi1);
			output.setPhiEnd(phi2);
			
			int64_t nPhiTot = numPlanesTot();
			// Tensor<double, 3> rOut(rVals.size(), zVals.size(), nPhiTot);
			// Tensor<double, 3> zOut(rVals.size(), zVals.size(), nPhiTot);
			auto rOut = mapToDevice(Tensor<double, 3>(rVals.size(), zVals.size(), nPhiTot), *device, true);
			auto zOut = mapToDevice(Tensor<double, 3>(rVals.size(), zVals.size(), nPhiTot), *device, true);
			
			auto rOutRef = rOut -> getHost();
			auto zOutRef = zOut -> getHost();
			
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
						
						rOutRef(iR, iZ, halfPoint - iPhi - 1) = rBwd;
						zOutRef(iR, iZ, halfPoint - iPhi - 1) = zBwd;
						lenOut(iR, iZ, halfPoint - iPhi - 1) = -lBwd;
						
						rOutRef(iR, iZ, halfPoint + iPhi + 1) = rFwd;
						zOutRef(iR, iZ, halfPoint + iPhi + 1) = zFwd;
						lenOut(iR, iZ, halfPoint + iPhi + 1) = lFwd;
					}
				}
			}
			
			rOut -> updateDevice();
			zOut -> updateDevice();
			
			for(auto iR : kj::indices(rVals)) {
				for(auto iZ : kj::indices(zVals)) {
					rOutRef(iR, iZ, halfPoint) = rVals[iR];
					zOutRef(iR, iZ, halfPoint) = zVals[iZ];
					lenOut(iR, iZ, halfPoint) = 0;
				}
			}
			
			writeTensor(rOutRef, output.getR());
			writeTensor(zOutRef, output.getZ());
			writeTensor(lenOut, output.getTraceLen());
			
			// Now we want to compute the inverse mapping
			
			// Compute grid bounds
			double rMax = -std::numeric_limits<double>::infinity();
			double rMin = -rMax;
			double zMax = rMax;
			double zMin = rMin;
			
			for(double r : output.getR().getData()) {
				if(r != r) continue;
				
				rMax = std::max(r, rMax);
				rMin = std::min(r, rMin);
			}
			
			for(double z : output.getZ().getData()) {
				if(z != z) continue;
				
				zMax = std::max(zMax, z);
				zMin = std::min(zMin, z);
			}
			
			// Prepare inverse mapping tensor
			
			// Note: These values are chosen arbitrarily at the moment
			const size_t nR = rVals.size();
			const size_t nZ = zVals.size();
			
			auto uVals = mapToDevice(Tensor<double, 3>(nR, nZ, nPhiTot), *device, true);
			auto vVals = mapToDevice(Tensor<double, 3>(nR, nZ, nPhiTot), *device, true);
			
			auto invOut = output.initInverse();
			invOut.setRMin(rMin);
			invOut.setRMax(rMax);
			invOut.setZMin(zMin);
			invOut.setZMax(zMax);
			
			Promise<void> kernelOp = FSC_LAUNCH_KERNEL(
				invertRflmKernel, *device,
				nR * nZ * nPhiTot,
				
				FSC_KARG(rOut, ALIAS_IN), FSC_KARG(zOut, ALIAS_IN),
				FSC_KARG(uVals, ALIAS_OUT), FSC_KARG(vVals, ALIAS_OUT),
				
				rMin, rMax, zMin, zMax
			);
			
			return kernelOp.then([this, uVals = mv(uVals), vVals = mv(vVals), invOut]() mutable {
				writeTensor(uVals -> getHost(), invOut.initU());
				writeTensor(vVals -> getHost(), invOut.initV());
			});
		}
		
		Promise<void> run() {
			return traceDirection(false)
			.then([this](auto cwResponse) {
				return traceDirection(true)
				.then([this, cwResponse = mv(cwResponse)](auto ccwResponse) {
					return processTraces(ccwResponse.getPoincareHits(), cwResponse.getPoincareHits());
				});
			});
		}
	};
	
	struct MapperImpl : public Mapper::Server {
		FLT::Client flt;
		KDTreeService::Client indexer;
		GeometryLib::Client geoLib;
		
		Own<DeviceBase> device;
		
		MapperImpl(FLT::Client flt, KDTreeService::Client indexer, GeometryLib::Client geoLib, DeviceBase& dev) :
			flt(mv(flt)), indexer(mv(indexer)), geoLib(mv(geoLib)), device(dev.addRef())
		{}
		
		Promise<void> computeRFLM(ComputeRFLMContext ctx) {
			auto params = ctx.getParams();
			auto planes = params.getMappingPlanes();
			
			KJ_REQUIRE(params.hasField(), "Must specify magnetic field");
			
			Temporary<ReversibleFieldlineMapping> result;
			result.setSurfaces(planes);
			result.setNPad(params.getNumPaddingPlanes());
			result.setNSym(params.getNSym());
			
			auto promiseBuilder = kj::heapArrayBuilder<Promise<void>>(planes.size());
			// Promise<void> computationPromise = READY_NOW;
			
			auto sections = result.initSections(planes.size());
			
			auto u0 = params.getU0();
			auto v0 = params.getV0();
			
			KJ_REQUIRE(hasMaximumOrdinal(params, 12), "You are trying to use features that this server version does not support");
			
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
				
				Shared<RFLMSectionTrace> trace(flt, params, section, *device);
				
				trace -> phi1 = planes[i1];
				trace -> phi2 = planes[i2];
				trace -> numTracingPlanes = params.getNumPlanes();
				trace -> numPaddingPlanes = params.getNumPaddingPlanes();
				trace -> rVals = params.getGridR();
				trace -> zVals = params.getGridZ();
				
				// Adjust if mapping has higher symmetry
				if(i2 == 0 && params.getNSym() > 1) {
					trace -> phi2 += 2 * fsc::pi / params.getNSym();
				}
				
				// KJ_DBG(trace -> phi1, trace -> phi2);
				
				totalWidth += trace -> computeWidth();
				
				/*computationPromise = computationPromise.then([trace]() mutable {
					return trace -> run();
				}).attach(trace.x());*/
				promiseBuilder.add(trace -> run().attach(cp(trace)));
			}
			
			KJ_REQUIRE(std::abs(totalWidth - 2 * pi / params.getNSym()) < pi, "Sections must be of non-zero width, and angles must be specified in counter-clockwise direction");
					
			auto computationPromise = kj::joinPromises(promiseBuilder.finish());
			return computationPromise.then([ctx = mv(ctx), result = mv(result)]() mutable {
				ctx.initResults().setMapping(getActiveThread().dataService().publish(mv(result)));
			});
		}
		
		Promise<void> mapMesh(Mesh::Reader in, Mesh::Builder out, uint64_t section, FSC_READER_MAPPING(::fsc, ReversibleFieldlineMapping) mapping, double threshold) {
			Tensor<double, 2> points;
			readTensor(in.getVertices(), points);
			
			size_t nPoints = points.dimension(1);
			
			Temporary<RFLMKernelData> kData;
			kData.initPhiValues(nPoints);
			kData.initStates(nPoints);
			kData.initReconstructionErrors(nPoints);
			
			FSC_BUILDER_MAPPING(::fsc, RFLMKernelData) kernelData =
				FSC_MAP_BUILDER(::fsc, RFLMKernelData, mv(kData), *device, true);
				
			kernelData -> updateStructureOnDevice();
			
			Promise<void> kernelTask = FSC_LAUNCH_KERNEL(
				mapInSectionKernel, *device,
				nPoints,
				
				section, FSC_KARG(mv(points), ALIAS_IN), kernelData, FSC_KARG(mapping, NOCOPY)
			);
			
			return kernelTask
			.then([data = mv(kernelData), in, out, nPoints, threshold]() mutable {
				out.setIndices(in.getIndices());
				if(in.isTriMesh()) out.setTriMesh();
				else if(in.isPolyMesh()) out.setPolyMesh(in.getPolyMesh());
				else { KJ_FAIL_REQUIRE("Unknown mesh type"); }
				
				
				RFLMKernelData::Reader mapResult = data -> getHost();
				
				// Check points for mapping failure
				auto indicesOut = out.getIndices();
				auto checkPoly = [&](uint64_t start, uint64_t end) {
					bool polyOk = true;
					
					for(auto iIdx : kj::range(start, end)) {
						auto idx = indicesOut[iIdx];
						double recError = mapResult.getReconstructionErrors()[idx];
						//double u = mapResult.getStates()[idx].getU();
						//double v = mapResult.getStates()[idx].getV();
						
						if(recError > threshold /*|| u < 0 || u > 1 || v < 0 || v > 1*/) {
							// KJ_DBG(idx, recError, u, v);
							polyOk = false;
						}
					}
					
					if(polyOk) return;
					
					// If the polygon is bad, turn it into degenerate
					//KJ_DBG("Clearing poly", start, end);
					for(auto iIdx : kj::range(start, end)) {
						// KJ_DBG(iIdx, indicesOut[iIdx]);
						indicesOut.set(iIdx, 0);
					}
				};
				
				if(out.isTriMesh()) {
					for(uint64_t i = 0; i < indicesOut.size(); i += 3)
						checkPoly(i, i + 3);
				} else if(out.isPolyMesh()) {
					auto pm = out.getPolyMesh();
					for(size_t i = 0; i + 1 < pm.size(); ++i) {
						checkPoly(pm[i], pm[i+1]);
					}
				} else { KJ_FAIL_REQUIRE("Unknown mesh type"); }
				
				Tensor<double, 2> points(3, nPoints);
				for(auto i : kj::range(0, nPoints)) {
					// Analogously to Phi, z, r we store Phi, v, u
					points(0, i) = mapResult.getStates()[i].getPhi();
					points(1, i) = mapResult.getStates()[i].getV();
					points(2, i) = mapResult.getStates()[i].getU(); 
				}
				writeTensor(points, out.getVertices());
			});
		}
		
		Promise<void> mapForSection(
			DataRef<MergedGeometry>::Client geoRef, GeometryMapping::SectionData::Builder out,
			FSC_READER_MAPPING(::fsc, ReversibleFieldlineMapping) mapping, uint64_t section,
			uint32_t nPhi, uint32_t nU, uint32_t nV, double threshold
		) {
			auto mappingData = mapping -> getHost();
			uint32_t nSections = mappingData.getSections().size();
			auto sectionData = mappingData.getSections()[section % nSections];
			
			double phiOffset = 2 * pi / mappingData.getNSym() * (section / nSections); // Rounding division inside bracket
			
			double phi1 = sectionData.getPhiStart() + phiOffset;
			double phi2 = sectionData.getPhiEnd() + phiOffset;
			
			while(phi2 <= phi1) {
				phi2 += 2 * pi;
			}
			
			out.setPhi1(phi1);
			out.setPhi2(phi2);
			
			// Fill out grid
			{
				auto g = out.getGrid();
				g.setXMin(phi1);
				g.setXMax(phi2);
				g.setNX(nPhi);
				
				g.setYMin(0);
				g.setYMax(1);
				g.setNY(nV);
				
				g.setZMin(0);
				g.setZMax(1);
				g.setNZ(nU);
			}
			
			// Unroll
			auto unrollRequest = geoLib.unrollRequest();
			unrollRequest.getGeometry().setMerged(geoRef);
			unrollRequest.getPhi1().setRad(phi1 - 1.5 * RFLM::SECTION_TOL);
			unrollRequest.getPhi2().setRad(phi2 + 1.5 * RFLM::SECTION_TOL);
			auto unrolledRef = unrollRequest.send().getRef();
			
			return getActiveThread().dataService().download(unrolledRef)
			.then([mapping = mapping -> addRef(), section, out = out.initGeometry(), threshold, this](auto mergedGeo) mutable {
				// Transform individual meshes
				auto in = mergedGeo.get();
				out.setTagNames(in.getTagNames());
				
				auto inEntries = in.getEntries();
				auto outEntries = out.initEntries(inEntries.size());
				
				auto meshTransforms = kj::heapArrayBuilder<Promise<void>>(inEntries.size());
				
				for(auto i : kj::indices(inEntries)) {
					auto eIn = inEntries[i];
					auto eOut = outEntries[i];
					eOut.setTags(eIn.getTags());
					
					meshTransforms.add(mapMesh(eIn.getMesh(), eOut.getMesh(), section, mapping -> addRef(), threshold));
				}
				
				return kj::joinPromisesFailFast(meshTransforms.finish()).attach(kj::cp(mergedGeo));
			})
			.then([section, merged = out.getGeometry(), grid = out.getGrid(), this]() {		
				// Index geometry over grid
				auto indexRequest = geoLib.indexRequest();
				indexRequest.setGrid(grid);
				indexRequest.getGeometry().setMerged(getActiveThread().dataService().publish(merged));
				
				auto dataRef = indexRequest.send().getIndexed().getData();
				return getActiveThread().dataService().download(dataRef);
			}).attach(thisCap())
			.then([section, out](auto localRef) mutable {
				out.setIndex(localRef.get());
			});
		}

		Promise<void> mapGeometry(MapGeometryContext ctx) override {
			return getActiveThread().dataService().download(ctx.getParams().getMapping())
			.then([ctx, this](auto mapping) mutable {
				auto devMapping = FSC_MAP_READER(
					::fsc, ReversibleFieldlineMapping,
					mapping, *device, true
				);
				devMapping -> updateStructureOnDevice();
				devMapping -> updateDevice();
				
				auto params = ctx.getParams();
				
				// Pre-merge geometry
				auto mergeRequest = geoLib.mergeRequest();
				mergeRequest.setNested(params.getGeometry());
				auto geoRef = mergeRequest.sendForPipeline().getRef();
				
				uint32_t nSectionsTot = mapping.get().getNSym() * mapping.get().getSections().size();
				uint32_t nSectionsCompute = nSectionsTot / params.getNSym();
				
				Temporary<GeometryMapping::MappingData> data;
				data.setNSym(params.getNSym());
				auto sections = data.initSections(nSectionsCompute);
				auto subComputations = kj::heapArrayBuilder<Promise<void>>(nSectionsCompute);
				
				for(auto i : kj::indices(sections)) {
					subComputations.add(
						mapForSection(geoRef, sections[i], devMapping -> addRef(), i, params.getNPhi(), params.getNU(), params.getNV(), params.getErrorThreshold())
					);
				}
				
				return kj::joinPromisesFailFast(subComputations.finish())
				.then([data = mv(data), ctx]() mutable {
					GeometryMapping::Builder result = ctx.getResults().getMapping();
					
					result.setData(getActiveThread().dataService().publish(mv(data)));
					result.setBase(ctx.getParams().getMapping());
				});
			});
		}
		
		Promise<void> getSectionGeometry(GetSectionGeometryContext ctx) override {
			return getActiveThread().dataService().download(ctx.getParams().getMapping().getData())
			.then([ctx](auto localRef) mutable {
				auto data = localRef.get();
				auto iSection = ctx.getParams().getSection();
				KJ_REQUIRE(iSection < data.getSections().size());
				
				auto section = data.getSections()[iSection];
				
				auto& ds = getActiveThread().dataService();
				
				auto geo = ctx.initResults().getGeometry();
				geo.setBase(ds.publish(section.getGeometry()));
				geo.setData(ds.publish(section.getIndex()));
				geo.setGrid(section.getGrid());
			});
		}
		
		Promise<void> geometryToFieldAligned(GeometryToFieldAlignedContext ctx) override {
			// Pre-merge geometry
			auto mergeRequest = geoLib.mergeRequest();
			mergeRequest.setNested(ctx.getParams().getGeometry());
			auto geoRef = mergeRequest.sendForPipeline().getRef();
			
			// Download merged geometry
			return getActiveThread().dataService().download(geoRef)
			.then([ctx, this](auto geo) mutable {
				
				// Download mapping
				return getActiveThread().dataService().download(ctx.getParams().getMapping())
				.then([ctx, this, geo](auto mapping) mutable {
					// Map mapping onto compute device
					auto devMapping = FSC_MAP_READER(
						::fsc, ReversibleFieldlineMapping,
						mapping, *device, true
					);
					devMapping -> updateStructureOnDevice();
					devMapping -> updateDevice();
					
					// Initialize output geometry
					Temporary<MergedGeometry> outputGeometry(geo.get());
					
					auto entries = geo.get().getEntries();
					auto entriesOut = outputGeometry.getEntries();
					
					auto subComputations = kj::heapArrayBuilder<Promise<void>>(entries.size());
					
					for(auto i : kj::indices(entries)) {
						auto eOut = entriesOut[i];
						auto meshOut = eOut.getMesh();
						
						Tensor<double, 2> initialVertexTensor;
						readTensor(entries[i].getMesh().getVertices(), initialVertexTensor);
						
						auto vertices = mapToDevice(kj::mv(initialVertexTensor), *device, true);
						
						Promise<void> subComputation = FSC_LAUNCH_KERNEL(
							toFieldAlignedKernel, *device, vertices -> getHost().dimension(1),
							
							ctx.getParams().getPhi0(), ctx.getParams().getR0(),
							FSC_KARG(devMapping, NOCOPY),
							vertices
						).then([keepAlive = vertices -> addRef(), meshOut, vertHost = vertices -> getHost()]() mutable {
							writeTensor(vertHost, meshOut.getVertices());
						});
						
						subComputations.add(kj::mv(subComputation));
					}
					
					return kj::joinPromisesFailFast(subComputations.finish())
					.then([ctx, og = kj::mv(outputGeometry)]() mutable {
						ctx.initResults().setGeometry(getActiveThread().dataService().publish(og.asReader()));
					});
				});
			});
		}
		
		Promise<void> geometryFromFieldAligned(GeometryFromFieldAlignedContext ctx) override {
			// Pre-merge geometry
			auto mergeRequest = geoLib.mergeRequest();
			mergeRequest.setNested(ctx.getParams().getGeometry());
			auto geoRef = mergeRequest.sendForPipeline().getRef();
			
			// Download merged geometry
			return getActiveThread().dataService().download(geoRef)
			.then([ctx, this](auto geo) mutable {
				
				// Download mapping
				return getActiveThread().dataService().download(ctx.getParams().getMapping())
				.then([ctx, this, geo](auto mapping) mutable {
					// Map mapping onto compute device
					auto devMapping = FSC_MAP_READER(
						::fsc, ReversibleFieldlineMapping,
						mapping, *device, true
					);
					devMapping -> updateStructureOnDevice();
					devMapping -> updateDevice();
					
					// Initialize output geometry
					Temporary<MergedGeometry> outputGeometry(geo.get());
					
					auto entries = geo.get().getEntries();
					auto entriesOut = outputGeometry.getEntries();
					
					auto subComputations = kj::heapArrayBuilder<Promise<void>>(entries.size());
					
					for(auto i : kj::indices(entries)) {
						auto eOut = entriesOut[i];
						auto meshOut = eOut.getMesh();
						
						Tensor<double, 2> initialVertexTensor;
						readTensor(entries[i].getMesh().getVertices(), initialVertexTensor);
						
						auto vertices = mapToDevice(kj::mv(initialVertexTensor), *device, true);
						
						Promise<void> subComputation = FSC_LAUNCH_KERNEL(
							fromFieldAlignedKernel, *device, vertices -> getHost().dimension(1),
							
							ctx.getParams().getPhi0(), ctx.getParams().getR0(),
							FSC_KARG(devMapping, NOCOPY),
							vertices
						).then([keepAlive = vertices -> addRef(), meshOut, vertHost = vertices -> getHost()]() mutable {
							writeTensor(vertHost, meshOut.getVertices());
						});
						
						subComputations.add(kj::mv(subComputation));
					}
					
					return kj::joinPromisesFailFast(subComputations.finish())
					.then([ctx, og = kj::mv(outputGeometry)]() mutable {
						ctx.initResults().setGeometry(getActiveThread().dataService().publish(og.asReader()));
					});
				});
			});
		}
	};
}

}

kj::Own<fsc::Mapper::Server> fsc::newMapper(FLT::Client flt, KDTreeService::Client indexer, GeometryLib::Client geoLib, DeviceBase& device) {
	return kj::heap<MapperImpl>(mv(flt), mv(indexer), mv(geoLib), device);
}
