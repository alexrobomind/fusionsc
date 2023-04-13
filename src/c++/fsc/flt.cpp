#include "flt-kernels.h"
#include "cudata.h"
#include "kernels.h"
#include "flt.h"

#include <algorithm>

#include <kj/vector.h>
// #include <capnp/serialize-text.h>
#include <fsc/flt.capnp.cu.h>

using namespace fsc;

namespace {
	
void validateField(ToroidalGrid::Reader grid, Float64Tensor::Reader data) {	
	auto shape = data.getShape();
	KJ_REQUIRE(shape.size() == 4);
	KJ_REQUIRE(shape[0] == grid.getNPhi());
	KJ_REQUIRE(shape[1] == grid.getNZ());
	KJ_REQUIRE(shape[2] == grid.getNR());
	KJ_REQUIRE(shape[3] == 3);
	
	KJ_REQUIRE(data.getData().size() == shape[0] * shape[1] * shape[2] * shape[3]);	
}

struct TraceCalculation {
	constexpr static size_t STEPS_PER_ROUND = 1000;
	constexpr static size_t EVENTBUF_SIZE = 2500;
	constexpr static size_t EVENTBUF_SIZE_NOGEO = 100;
		
	struct Round {
		DeviceMappingType<CuTypedMessageBuilder<
			FLTKernelData, cu::FLTKernelData
		>> kernelData;
		
		DeviceMappingType<CuTypedMessageBuilder<
			FLTKernelRequest, cu::FLTKernelRequest
		>> kernelRequest;
		
		kj::Vector<size_t> participants;
		
		size_t upperBound;
	};
	
	Own<DeviceBase> device;
	Tensor<double, 2> positions;
	
	DeviceMappingType<Own<TensorMap<const Tensor<double, 4>>>> field;
	
	Temporary<FLTKernelRequest> request;
	kj::Vector<Round> rounds;
	
	uint32_t ROUND_STEP_LIMIT = 1000000;
		
	DeviceMappingType<CuTypedMessageBuilder<
		IndexedGeometry, cu::IndexedGeometry
	>> indexedGeometry;
	
	DeviceMappingType<CuTypedMessageReader<
		IndexedGeometry::IndexData, cu::IndexedGeometry::IndexData
	>> indexData;
	
	DeviceMappingType<CuTypedMessageReader<
		MergedGeometry, cu::MergedGeometry
	>> geometryData;
	
	DeviceMappingType<CuTypedMessageReader<
		FieldlineMapping, cu::FieldlineMapping
	>> mappingData;
		
	TraceCalculation(DeviceBase& device,
		Temporary<FLTKernelRequest>&& newRequest, Own<TensorMap<const Tensor<double, 4>>> newField, Tensor<double, 2> positions,
		IndexedGeometry::Reader geometryIndex, Maybe<LocalDataRef<IndexedGeometry::IndexData>> indexData, Maybe<LocalDataRef<MergedGeometry>> geometryData,
		Maybe<LocalDataRef<FieldlineMapping>> mappingData
	) :
		device(device.addRef()),
		positions(mv(positions)),
		
		field(mapToDevice(mv(newField), device, true)),
		
		request(mv(newRequest)),
		
		indexedGeometry(mapToDevice(
			cuBuilder<IndexedGeometry, cu::IndexedGeometry>(
				Temporary<IndexedGeometry>(geometryIndex)
			),
			device, true
		)),
		indexData(mapToDevice(
			cuReader<IndexedGeometry::IndexData, cu::IndexedGeometry::IndexData>(indexData),
			device, true
		)),
		geometryData(mapToDevice(
			cuReader<MergedGeometry, cu::MergedGeometry>(geometryData),
			device, true
		)),
		mappingData(mapToDevice(
			cuReader<FieldlineMapping, cu::FieldlineMapping>(mappingData),
			device, true
		))
	{		
		if(request.getServiceRequest().getRngSeed() == 0) {
			uint64_t seed;
			getActiveThread().rng().randomize(kj::ArrayPtr<unsigned char>(reinterpret_cast<unsigned char*>(&seed), sizeof(decltype(seed))));
			
			request.getServiceRequest().setRngSeed(seed);
		}
	}
	
	// Prepares the memory structure for a round
	Round& prepareRound(size_t nParticipants) {
		Round round;
		
		Temporary<FLTKernelData> kDataIn;
		
		auto data = kDataIn.initData(nParticipants);
		for(size_t i = 0; i < nParticipants; ++i) {
			
			data[i].initState();
			auto events = data[i].initEvents(/*request.getServiceRequest().hasGeometry() ? EVENTBUF_SIZE : EVENTBUF_SIZE_NOGEO*/EVENTBUF_SIZE);
			
			for(auto event : events)
				event.initLocation(3);
		}
		
		round.kernelData = mapToDevice(
			cuBuilder<FLTKernelData, cu::FLTKernelData>(mv(kDataIn)),
			*device, true
		);
		round.kernelRequest = mapToDevice(
			cuBuilder<FLTKernelRequest, cu::FLTKernelRequest>(
				Temporary<FLTKernelRequest>(request.asReader())
			),
			*device, true
		);
		
		round.participants.reserve(nParticipants);
		
		return rounds.add(mv(round));
	}
	
	Round& setupInitialRound() {
		KJ_REQUIRE(positions.dimension(0) == 3);
		
		KJ_REQUIRE(rounds.size() == 0, "Internal error");
		
		const size_t nParticipants = positions.dimension(1);
		Round& round = prepareRound(nParticipants);
		
		round.participants.addAll(kj::range<size_t>(0, nParticipants));	
		
		std::mt19937_64 seedGenerator(request.getServiceRequest().getRngSeed());
		
		auto data = round.kernelData -> getHost().getData();
		for(size_t i = 0; i < nParticipants; ++i) {
			auto state = data[i].initState();
			
			auto pos = state.initPosition(3);
			for(unsigned char iDim = 0; iDim < 3; ++iDim)
				pos.set(iDim, positions(iDim, i));
			
			state.setPhi0(std::atan2(pos[1], pos[0]));
			state.setForward(request.getServiceRequest().getForward());
			
			fsc::MT19937::seed((uint32_t) seedGenerator(), state.getRngState());
		}
		
		round.kernelRequest -> getHost().getServiceRequest().setStepLimit(
			request.getServiceRequest().getStepLimit() != 0 ? std::min(request.getServiceRequest().getStepLimit(), ROUND_STEP_LIMIT) : ROUND_STEP_LIMIT
		);
		
		// Initialize device memory
		indexedGeometry -> updateDevice();
		indexData -> updateDevice();
		geometryData -> updateDevice();
		mappingData -> updateDevice();
		field -> updateDevice();
		
		return round;
	}
	
	Round& setupFollowupRound() {
		KJ_REQUIRE(rounds.size() > 0, "Internal error");
		
		// Check previous round
		Round* prevRound = &(rounds[rounds.size() - 1]);
		
		// Count unfinished participants
		kj::Vector<size_t> unfinished;
		auto kDataIn = prevRound -> kernelData -> getHost().getData();
		
		KJ_REQUIRE(kDataIn.size() == prevRound -> participants.size());
		
		for(size_t i = 0; i < kDataIn.size(); ++i) {
			if(!isFinished(kDataIn[i])) unfinished.add(i);
		}
		
		KJ_DBG(prevRound -> participants, unfinished);
		
		KJ_REQUIRE(unfinished.size() > 0, "Internal error");
		
		uint32_t maxSteps = 0;
		
		Round& newRound = prepareRound(unfinished.size());
		prevRound = &(rounds[rounds.size() - 2]);
		
		auto kDataOut = newRound.kernelData -> getHost().getData();
		for(size_t i = 0; i < unfinished.size(); ++i) {
			// KJ_DBG(i, unfinished[i], prevRound -> participants.size());
			
			newRound.participants.add(prevRound -> participants[unfinished[i]]);
			
			auto entryOut = kDataOut[i];
			auto entryIn  = kDataIn[unfinished[i]];
			
			entryOut.setState(entryIn.getState());
			entryOut.getState().setEventCount(0);
			
			maxSteps = std::max(maxSteps, entryOut.getState().getNumSteps());
			// KJ_DBG(entryOut.getState());
		}
		
		if(request.getServiceRequest().getStepLimit() != 0) {
			newRound.kernelRequest -> getHost().getServiceRequest().setStepLimit(
				std::min(request.getServiceRequest().getStepLimit(), maxSteps + ROUND_STEP_LIMIT)
			);
		} else {
			newRound.kernelRequest -> getHost().getServiceRequest().setStepLimit(maxSteps + ROUND_STEP_LIMIT);
		}
		
		KJ_DBG("Relaunching", maxSteps, ROUND_STEP_LIMIT);
		
		return newRound;
	}
	
	Round& setupRound() {
		if(rounds.size() == 0)
			return setupInitialRound();
		else
			return setupFollowupRound();
	}
		
	
	Promise<void> startRound(Round& r) {
		r.kernelData -> updateStructureOnDevice();
		r.kernelRequest -> updateStructureOnDevice();
		
		return FSC_LAUNCH_KERNEL(
			fltKernel, *device,
			r.kernelData -> getHost().getData().size(),
			
			r.kernelData, FSC_KARG(field, NOCOPY), r.kernelRequest,
			FSC_KARG(geometryData, NOCOPY), FSC_KARG(indexedGeometry, NOCOPY), FSC_KARG(indexData, NOCOPY),
			FSC_KARG(mappingData, NOCOPY)
		);
	}
	
	bool isFinished(FLTKernelData::Entry::Reader entry) {
		auto stopReason = entry.getStopReason();
		
		if(stopReason == FLTStopReason::UNKNOWN) {
			KJ_FAIL_REQUIRE("Kernel stopped for unknown reason");
		}
		
		if(stopReason == FLTStopReason::EVENT_BUFFER_FULL)
			return false;
		
		if(stopReason == FLTStopReason::STEP_LIMIT && (request.getServiceRequest().getStepLimit() == 0 || entry.getState().getNumSteps() < request.getServiceRequest().getStepLimit()))
			return false;
		
		return true;
	}
	
	bool isFinished(Round& r) {
		for(auto kDatum : r.kernelData -> getHost().getData()) {
			if(!isFinished(kDatum))
				return false;
		}
		
		return true;
	}
	
	Promise<void> runRound() {
		auto& round = setupRound();
		return startRound(round);
	}		
	
	Promise<void> run() {
		if(rounds.size() == 0) {
			return runRound().then([this]() { return run(); });
		}
			
		auto& round = rounds[rounds.size() - 1];
			
		if(isFinished(round)) {
			KJ_DBG("Trace Finished");
			return READY_NOW;
		}
		
		return runRound().then([this]() { return run(); });
	}
	
	Temporary<FLTKernelData::Entry> consolidateRuns(size_t participantIndex) {
		const size_t nRounds = rounds.size();
		auto participantIndices = kj::heapArray<Maybe<size_t>>(nRounds);
		
		for(size_t i = 0; i < nRounds; ++i) {
			auto& round = rounds[i];
			auto range = std::equal_range(round.participants.begin(), round.participants.end(), participantIndex);
			
			if(range.first != range.second)
				participantIndices[i] = range.first - round.participants.begin();
			else
				participantIndices[i] = nullptr;
		}
				
		size_t nEvents = 0;
		for(size_t i = 0; i < nRounds; ++i) {
			KJ_IF_MAYBE(pIdx, participantIndices[i]) {
				auto kData = rounds[i].kernelData -> getHost().getData()[*pIdx];
				size_t eventCount = kData.getState().getEventCount();
				
				nEvents += eventCount;
			}
		}
		
		Temporary<FLTKernelData::Entry> result;
		auto eventsOut = result.initEvents(nEvents);
		
		size_t iEvent = 0;
		for(size_t i = 0; i < nRounds; ++i) {
			KJ_IF_MAYBE(pIdx, participantIndices[i]) {
				auto kData = rounds[i].kernelData -> getHost().getData()[*pIdx];
				size_t eventCount = kData.getState().getEventCount();
				
				auto eventsIn = kData.getEvents();
				KJ_REQUIRE(eventCount <= eventsIn.size());
				
				for(size_t iEvtIn = 0; iEvtIn < eventCount; ++iEvtIn) {
					eventsOut.setWithCaveats(iEvent++, eventsIn[iEvtIn]);
				}
				
				result.setStopReason(kData.getStopReason());
				result.setState(kData.getState());
			}
		}
		
		KJ_REQUIRE(isFinished(result.asReader()));
		
		return result;
	}
};

struct FLTImpl : public FLT::Server {
	Own<DeviceBase> device;
	
	FLTImpl(Own<DeviceBase> device) : device(mv(device)) {}
	
	Promise<void> trace(TraceContext ctx) override {
		// ctx.allowCancellation();
		auto request = ctx.getParams();
		
		// Request validation
		for(auto plane : request.getPlanes()) {
			if(plane.hasCenter()) {
				KJ_REQUIRE(plane.getCenter().size() == 3, "Centers must be 3-dimensional");
			}
			if(plane.getOrientation().isNormal()) {
				KJ_REQUIRE(plane.getOrientation().getNormal().size() == 3, "Plane normals must be 3D", plane);
			}
			if(plane.getOrientation().isPhi()) {
				KJ_REQUIRE(!plane.hasCenter(), "Unimplemented: Phi planes are only supported when center is not specified (assuming 0, 0, 0)");
			}
		}
		
		auto& dataService = getActiveThread().dataService();
		
		auto isNull = [&](capnp::Capability::Client c) {
			return capnp::ClientHook::from(mv(c))->isNull();
		};
		
		auto indexDataRef = request.getGeometry().getData();
		auto geoDataRef = request.getGeometry().getBase();
		auto mappingDataRef = request.getMapping();
			
		KJ_DBG("Initiating download processes");
		
		Promise<LocalDataRef<Float64Tensor>> downloadField = dataService.download(request.getField().getData()).eagerlyEvaluate(nullptr);
		Promise<Maybe<LocalDataRef<IndexedGeometry::IndexData>>> downloadIndexData = dataService.downloadIfNotNull(indexDataRef);
		Promise<Maybe<LocalDataRef<MergedGeometry>>> downloadGeometryData =	dataService.downloadIfNotNull(geoDataRef).eagerlyEvaluate(nullptr);
		Promise<Maybe<LocalDataRef<FieldlineMapping>>> downloadMappingData = dataService.downloadIfNotNull(mappingDataRef).eagerlyEvaluate(nullptr);
		KJ_DBG("Waiting download processes");
		
		// I WANT COROUTINES -.-
		
		return downloadIndexData.then([ctx, request, downloadGeometryData = mv(downloadGeometryData), downloadField = mv(downloadField), downloadMappingData = mv(downloadMappingData), this](Maybe<LocalDataRef<IndexedGeometry::IndexData>> indexData) mutable {
		return downloadGeometryData.then([ctx, request, indexData = mv(indexData), downloadField = mv(downloadField), downloadMappingData = mv(downloadMappingData), this](Maybe<LocalDataRef<MergedGeometry>> geometryData) mutable {
		return downloadField.then([ctx, request, indexData = mv(indexData), geometryData = mv(geometryData), downloadMappingData = mv(downloadMappingData), this](LocalDataRef<Float64Tensor> fieldData) mutable {
		return downloadMappingData.then([ctx, request, indexData = mv(indexData), geometryData = mv(geometryData), fieldData = mv(fieldData), this](Maybe<LocalDataRef<FieldlineMapping>> mappingData) mutable {
			
			KJ_DBG("All required data downloaded");
			
			// Extract kernel request
			Temporary<FLTKernelRequest> kernelRequest;
			/*kernelRequest.setPhiPlanes(request.getPoincarePlanes());
			kernelRequest.setTurnLimit(request.getTurnLimit());
			kernelRequest.setCollisionLimit(request.getCollisionLimit());
			kernelRequest.setDistanceLimit(request.getDistanceLimit());
			kernelRequest.setStepLimit(request.getStepLimit());
			kernelRequest.setStepSize(request.getStepSize());
			kernelRequest.setGrid(request.getField().getGrid());*/
			kernelRequest.setServiceRequest(request);
			
			// Extract field data
			auto field = mapTensor<Tensor<double, 4>>(fieldData.get());
			
			// Extract positions
			auto inStartPoints = request.getStartPoints();
			auto startPointShape = inStartPoints.getShape();
			KJ_REQUIRE(startPointShape.size() >= 1, "Start points must have at least 1 dimension");
			KJ_REQUIRE(startPointShape[0] == 3, "First dimension of start points must have size 3");
			
			int64_t nStartPoints = 1;
			for(size_t i = 1; i < startPointShape.size(); ++i)
				nStartPoints *= startPointShape[i];
			
			// Reshape start points into linear shape
			Temporary<Float64Tensor> reshapedStartPoints;
			reshapedStartPoints.setData(inStartPoints.getData());
			{
				reshapedStartPoints.setShape({3, (uint64_t) nStartPoints});
				// shape[0] = 3; shape[1] = nStartPoints;
			}			
			
			// Try to map underlying tensor and load it into memory
			Tensor<double, 2> positions = mapTensor<Tensor<double, 2>>(reshapedStartPoints.asReader())
				-> shuffle(Eigen::array<int, 2>{1, 0});
			
			// KJ_UNIMPLEMENTED("Load mapping data");
			
			auto calc = heapHeld<TraceCalculation>(
				*device, mv(kernelRequest), mv(field), mv(positions),
				request.getGeometry(), mv(indexData), geometryData,
				mappingData
			);
			
			return calc->run()
			.then([ctx, calc, request, startPointShape, nStartPoints, geometryData = mv(geometryData)]() mutable {
				int64_t nTurns = 0;
				
				auto resultBuilder = kj::heapArrayBuilder<Temporary<FLTKernelData::Entry>>(nStartPoints);
				for(size_t i = 0; i < nStartPoints; ++i)
					resultBuilder.add(calc->consolidateRuns(i));
				
				auto kData = resultBuilder.finish();
								
				for(auto& entry : kData) {
					nTurns = std::max(nTurns, (int64_t) entry.getState().getTurnCount() + 1);
				}
				
				if(request.getTurnLimit() > 0)
					nTurns = std::min(nTurns, (int64_t) request.getTurnLimit());
				
				int64_t nSurfs = request.getPlanes().size();
				
				auto results = ctx.getResults();
				results.setNTurns(nTurns);
				
				Tensor<double, 4> pcCuts(nTurns, nStartPoints, nSurfs, 5);
				pcCuts.setConstant(std::numeric_limits<double>::quiet_NaN());
				
				Tensor<double, 2> endPoints(nStartPoints, 4);
					
				{
					auto stopReasons = results.initStopReasons();
					auto shape = stopReasons.initShape(startPointShape.size() - 1);
					for(auto i : kj::indices(shape))
						shape.set(i, startPointShape[i+1]);
				}
				
				KJ_IF_MAYBE(pGeometryData, geometryData) {
					auto geometry = pGeometryData -> get();
					
					auto tagNames = geometry.getTagNames();
					auto outTagNames = results.initTagNames(tagNames.size());
					for(auto i : kj::indices(outTagNames))
						outTagNames.set(i, tagNames[i]);
					
					{
						auto endTags = results.initEndTags();
						auto shape = endTags.initShape(startPointShape.size());
						
						shape.set(0, tagNames.size());
						for(auto i : kj::range(1, startPointShape.size()))
							shape.set(i, startPointShape[i]);
					}
					results.getEndTags().initData(geometry.getTagNames().size() * nStartPoints);
				} else {
					auto endTags = results.initEndTags();
					auto shape = endTags.initShape(startPointShape.size());
						
					shape.set(0, 0);
					for(auto i : kj::range(1, startPointShape.size()))
						shape.set(i, startPointShape[i]);
				}
				
				auto stopReasonData = results.getStopReasons().initData(nStartPoints);
				auto endTagData = results.getEndTags().getData();
				
				size_t nRecorded = 0;
				
				for(int64_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
					auto entry = kData[iStartPoint].asReader();
					auto state = entry.getState();
					auto events = entry.getEvents();
								
					int64_t iTurn = 0;
					
					kj::Vector<double> backwardLCs(events.size());
					Maybe<FLTKernelEvent::Reader> lastCollision;
					{
						for(auto evt : events) {
							if(evt.isGeometryHit()) {
								lastCollision = evt;
							}
							
							KJ_IF_MAYBE(pLastCollision, lastCollision) {
								backwardLCs.add(evt.getDistance() - pLastCollision -> getDistance());
							} else {
								backwardLCs.add(-evt.getDistance());
							}
						}
					}
					
					kj::Vector<double> forwardLCs;
					forwardLCs.resize(events.size());
					{
						Maybe<FLTKernelEvent::Reader> nextCollision;
						for(auto i : kj::indices(events)) {
							auto iEvt = events.size() - 1 - i;
							auto evt = events[iEvt];
							
							if(evt.isGeometryHit()) {
								nextCollision = evt;
							}
							
							KJ_IF_MAYBE(pNextCollision, nextCollision) {
								forwardLCs[iEvt] = pNextCollision -> getDistance() - evt.getDistance();
							} else {
								forwardLCs[iEvt] = -(state.getDistance() - evt.getDistance());
							}
						}
					}
					
					size_t recordedForThis = 0;
					
					for(auto iEvt : kj::indices(events)) {
						auto evt = events[iEvt];
						
						KJ_REQUIRE(!evt.isNotSet(), "Internal error: Event not set", iStartPoint, iEvt);
						
						// KJ_DBG(evt);
												
						if(evt.isNewTurn()) {
							iTurn = evt.getNewTurn();
						} else if(evt.isPhiPlaneIntersection()) {
							auto ppi = evt.getPhiPlaneIntersection();
							// KJ_DBG(iTurn, ppi.getPlaneNo());
							
							auto loc = evt.getLocation();
							for(int64_t iDim = 0; iDim < 3; ++iDim) {
								pcCuts(iTurn, iStartPoint, ppi.getPlaneNo(), iDim) = loc[iDim];
							}
							pcCuts(iTurn, iStartPoint, ppi.getPlaneNo(), 3) = forwardLCs[iEvt];
							pcCuts(iTurn, iStartPoint, ppi.getPlaneNo(), 4) = backwardLCs[iEvt];
						} else if(evt.isRecord()) {
							++recordedForThis;
						}
					}
					
					nRecorded = kj::max(nRecorded, recordedForThis);
					
					auto stopLoc = state.getPosition();
					for(int i = 0; i < 3; ++i) {
						endPoints(iStartPoint, i) = stopLoc[i];
					}
					endPoints(iStartPoint, 3) = state.getDistance();
					
					auto stopReason = entry.getStopReason();
					KJ_REQUIRE(stopReason != FLTStopReason::UNKNOWN, "Internal error: Unknown stop reason encountered", iStartPoint);
					KJ_REQUIRE(stopReason != FLTStopReason::EVENT_BUFFER_FULL, "Internal error: Invalid stop reason EVENT_BUFFER_FULL", iStartPoint);
					stopReasonData.set(iStartPoint, stopReason);
					
					if(stopReason == FLTStopReason::COLLISION_LIMIT) {
						KJ_IF_MAYBE(pLastCollision, lastCollision) {
							auto& lastCollision = *pLastCollision;
							
							uint32_t meshIdx = lastCollision.getGeometryHit().getMeshIndex();
							
							KJ_IF_MAYBE(pGeometryData, geometryData) {
								auto geometry = pGeometryData->get();
								auto tagValues = geometry.getEntries()[meshIdx].getTags();
								
								for(auto iTag : kj::indices(tagValues)) {
									endTagData.setWithCaveats(iTag * nStartPoints + iStartPoint, tagValues[iTag]);
								}
							} else {
								KJ_FAIL_REQUIRE("Internal error: Stop reason was COLLISION_LIMIT but geometry not set");
							}
						} else {
							KJ_FAIL_REQUIRE("Internal error: Stop reason was COLLISION_LIMIT but no collision recorded in event buffer", iStartPoint);
						}
					}
				}
				writeTensor(pcCuts, results.getPoincareHits());
				writeTensor(endPoints, results.getEndPoints());
				
				if(nRecorded > 0) {
					Tensor<double, 3> fieldLines(nRecorded, nStartPoints, 3);
					fieldLines.setConstant(std::nan(""));
					
					Tensor<double, 2> fieldStrengths(nRecorded, nStartPoints);
					fieldStrengths.setConstant(std::nan(""));
					
					for(auto iStartPoint : kj::range(0, nStartPoints)) {
						auto entry = kData[iStartPoint].asReader();
						auto events = entry.getEvents();
						
						int64_t iRecord = 0;
						for(auto evt : events) {
							// Process only "record" events
							if(!evt.isRecord())
								continue;
							
							auto loc = evt.getLocation();
							for(int32_t iDim = 0; iDim < 3; ++iDim)
								fieldLines(iRecord, iStartPoint, iDim) = loc[iDim];
							
							fieldStrengths(iRecord, iStartPoint) = evt.getRecord().getFieldStrength();
							
							++iRecord;
						}
					}
					
					writeTensor(fieldLines, results.getFieldLines());
					writeTensor(fieldStrengths, results.getFieldStrengths());
				}
				
				auto pcHitsShape = results.getPoincareHits().initShape(startPointShape.size() + 2);
				pcHitsShape.set(0, 5);
				pcHitsShape.set(1, nSurfs);
				for(int i = 0; i < startPointShape.size() - 1; ++i)
					pcHitsShape.set(i + 2, startPointShape[i + 1]);
				pcHitsShape.set(startPointShape.size() + 1, nTurns);
				
				auto fieldLinesShape = results.getFieldLines().initShape(startPointShape.size() + 1);
				for(auto i : kj::indices(startPointShape))
					fieldLinesShape.set(i, startPointShape[i]);
				fieldLinesShape.set(startPointShape.size(), nRecorded);
				
				auto fieldStrengthsShape = results.getFieldStrengths().initShape(startPointShape.size());
				for(auto i : kj::range(0, startPointShape.size() - 1)) {
					fieldStrengthsShape.set(i, startPointShape[i + 1]);
				}
				fieldStrengthsShape.set(startPointShape.size() - 1, nRecorded);

				results.getEndPoints().setShape(startPointShape);
				results.getEndPoints().getShape().set(0, 4);
			}).attach(calc.x());
		});
		});
		});
		}).attach(thisCap());
	}
	
	Promise<void> findAxis(FindAxisContext ctx) {
		struct IterResult {
			double r;
			double z;
		};
		
		auto params = ctx.getParams();
		auto xyz = params.getStartPoint();
		
		KJ_REQUIRE(xyz.size() == 3);
		
		double phi = atan2(xyz[1], xyz[0]);
		
		auto performIteration = [this, params, phi](IterResult iter) mutable {
			double r = iter.r;
			double z = iter.z;
			
			// Trace from starting position
			auto req = thisCap().traceRequest();
			auto sp = req.initStartPoints();
			
			sp.setShape({3});
			sp.setData({r * cos(phi), r * sin(phi), z});
			req.setTurnLimit(params.getNTurns());
			req.setStepSize(params.getStepSize());
			req.setField(params.getField());
			
			auto planes = req.initPlanes(1);
			planes[0].getOrientation().setPhi(phi);
			
			return req.send()
			.then([phi](capnp::Response<FLTResponse> response) mutable {
				// After tracing, take mean of points
				Tensor<double, 3> result;
				readTensor(response.getPoincareHits(), result);
				KJ_REQUIRE(result.dimension(1) == 1);
				KJ_REQUIRE(result.dimension(2) == 5);
				
				Tensor<double, 1> mean = result.mean(Eigen::array<int, 2>{0, 1});
				double x = mean[0];
				double y = mean[1];
				double z = mean[2];
				
				return IterResult { sqrt(x * x + y * y), z };
			});
		};
		
		auto traceAxis = [this, ctx, params, phi](IterResult iter) mutable {
			double r = iter.r;
			double z = iter.z;
			
			auto req = thisCap().traceRequest();
			auto sp = req.initStartPoints();
			sp.setShape({3});
			sp.setData({r * cos(phi), r * sin(phi), z});
			
			// Copy position into output
			ctx.getResults().setPos(sp.getData());
			
			req.setTurnLimit(1);
			req.setStepSize(params.getStepSize());
			req.setField(params.getField());
			
			req.setRecordEvery(1);
			
			return req.send()
			.then([ctx](capnp::Response<FLTResponse> response) mutable {
				auto axis = response.getFieldLines();
				
				ctx.getResults().setAxis(response.getFieldLines());
			});
		};
		
		double r = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
		double z = xyz[2];
		
		Promise<IterResult> nIterResult = IterResult { r, z };
		
		for(auto iteration : kj::range(0, params.getNIterations())) {
			nIterResult = nIterResult.then(cp(performIteration));
		}
		
		return nIterResult.then(mv(traceAxis));
	}
};
	
}

namespace fsc {	
	// TODO: Make this accept data service instead	
	FLT::Client newFLT(Own<DeviceBase> device) {
		return kj::heap<FLTImpl>(mv(device));
	}
}