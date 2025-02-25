#include "flt-kernels.h"

#include "kernels/launch.h"
#include "kernels/message.h"
#include "kernels/tensor.h"
#include "kernels/karg.h"

#include "flt.h"
#include "tensor.h"
#include "random-kernels.h"
#include "nudft.h"

#include <algorithm>
#include <limits>

#include <kj/vector.h>
// #include <capnp/serialize-text.h>
#include <fsc/flt.capnp.cu.h>

using namespace fsc;

namespace {

std::array<double, 3> getEventLocation(FLTKernelEvent::Reader e) {
	return {e.getX(), e.getY(), e.getZ()};
}
	
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
		FSC_BUILDER_MAPPING(::fsc, FLTKernelData) kernelData;
		FSC_BUILDER_MAPPING(::fsc, FLTKernelRequest) kernelRequest;
		
		kj::Vector<size_t> participants;
		size_t upperBound;
	};
	
	Own<DeviceBase> device;
	Tensor<double, 2> positions;
	
	DeviceMappingType<Own<TensorMap<const Tensor<double, 4>>>> field;
	
	Temporary<FLTKernelRequest> request;
	kj::Vector<Round> rounds;
	
	uint32_t ROUND_STEP_LIMIT = 1000000;
	
	FSC_BUILDER_MAPPING(::fsc, IndexedGeometry) indexedGeometry;
	
	FSC_READER_MAPPING(::fsc, IndexedGeometry::IndexData) indexData;
	FSC_READER_MAPPING(::fsc, MergedGeometry) geometryData;
	FSC_READER_MAPPING(::fsc, ReversibleFieldlineMapping) mappingData;
	FSC_READER_MAPPING(::fsc, GeometryMapping::MappingData) geoMappingData;

	Temporary<FLTConfig> config;
	
	uint32_t eventBufferSize = 0;
		
	TraceCalculation(DeviceBase& device,
		Temporary<FLTKernelRequest>&& newRequest, Own<TensorMap<const Tensor<double, 4>>> newField, Tensor<double, 2> nPositions,
		IndexedGeometry::Reader geometryIndex, Maybe<LocalDataRef<IndexedGeometry::IndexData>> indexData, Maybe<LocalDataRef<MergedGeometry>> geometryData,
		Maybe<LocalDataRef<ReversibleFieldlineMapping>> mappingData, Maybe<LocalDataRef<GeometryMapping::MappingData>> geoMappingData,
		FLTConfig::Reader config
	) :
		device(device.addRef()),
		positions(mv(nPositions)),
		
		field(mapToDevice(mv(newField), device, true)),
		
		request(mv(newRequest)),
		
		indexedGeometry(FSC_MAP_BUILDER(::fsc, IndexedGeometry, Temporary<IndexedGeometry>(geometryIndex), device, true)),
		
		indexData(FSC_MAP_READER(::fsc, IndexedGeometry::IndexData, indexData, device, true)),
		geometryData(FSC_MAP_READER(::fsc, MergedGeometry, geometryData, device, true)),
		
		mappingData(FSC_MAP_READER(::fsc, ReversibleFieldlineMapping, mappingData, device, true)),
		geoMappingData(FSC_MAP_READER(::fsc, GeometryMapping::MappingData, geoMappingData, device, true)),
		config(config)
	{		
		if(request.getServiceRequest().getRngSeed() == 0) {
			uint64_t seed;
			getActiveThread().rng().randomize(kj::ArrayPtr<unsigned char>(reinterpret_cast<unsigned char*>(&seed), sizeof(decltype(seed))));
			
			request.getServiceRequest().setRngSeed(seed);
		}
		
		KJ_REQUIRE(request.getServiceRequest().getForwardDirection().which() < 2, "Unknown forward direction type specified. This likely means that this library version is too old to understand your request");
		
		eventBufferSize = computeBufferSize(config.getEventBuffer(), positions.dimension(1));
	}
	
	uint32_t computeBufferSize(FLTConfig::EventBuffer::Reader config, uint32_t nPoints) {
		uint32_t wordBudget = config.getTargetTotalMb() * 1024 * 128; // 1024 kB per MB, 128 words per MB
		
		// We need one word per process for list headers
		if(wordBudget < nPoints)
			return 0;
		wordBudget -= nPoints;
		
		auto structSchema = capnp::StructSchema::from<FLTKernelEvent>();
		auto structInfo = structSchema.getProto().getStruct();
		uint32_t structSizeInWords = structInfo.getDataWordCount() + structInfo.getPointerCount();
		
		uint32_t size = wordBudget / (structSizeInWords * nPoints);

		KJ_LOG(INFO, "Buffer size estimation", nPoints, size, wordBudget, structSizeInWords, config);
		
		if(size < config.getMinSize())
			return config.getMinSize();
		
		if(size > config.getMaxSize())
			return config.getMaxSize();
		
		return size;
	}
	
	// Prepares the memory structure for a round
	Round& prepareRound(size_t nParticipants) {
		Round round;
		
		Temporary<FLTKernelData> kDataIn;
		
		auto data = kDataIn.initData(nParticipants);
		for(size_t i = 0; i < nParticipants; ++i) {
			
			data[i].initState();
			auto events = data[i].initEvents(eventBufferSize);
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
			
			double phi0 = std::atan2(pos[1], pos[0]);
			if(phi0 < 0) phi0 += 2 * pi;
			
			state.setPhi0(phi0);
			state.setPhi(phi0);
			state.setForward(request.getServiceRequest().getForward());
			
			fsc::MT19937::seed((uint32_t) seedGenerator(), state.getRngState());
			
			state.setStepSize(request.getServiceRequest().getStepSize());
		}
		
		round.kernelRequest -> getHost().getServiceRequest().setStepLimit(
			request.getServiceRequest().getStepLimit() != 0 ? std::min(request.getServiceRequest().getStepLimit(), ROUND_STEP_LIMIT) : ROUND_STEP_LIMIT
		);
		
		// Initialize device memory
		indexedGeometry -> updateDevice();
		indexData -> updateDevice();
		geometryData -> updateDevice();
		mappingData -> updateDevice();
		geoMappingData -> updateDevice();
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

		bool canIncreaseBufferSize = eventBufferSize < config.getEventBuffer().getMaxSize();
		bool increaseBufferSize = false;
		for(size_t i = 0; i < kDataIn.size(); ++i) {
			if(kDataIn[i].getStopReason() == FLTStopReason::COULD_NOT_STEP) {
				if(canIncreaseBufferSize)
					increaseBufferSize = true;
				else
					continue;
			}

			if(!isFinished(kDataIn[i])) unfinished.add(i);
		}

		if(increaseBufferSize) {
			auto oldBufferSize = eventBufferSize;
			eventBufferSize = std::min(2 * eventBufferSize, config.getEventBuffer().getMaxSize());

			KJ_LOG(INFO, "Growing event buffer size", oldBufferSize, eventBufferSize);
		}
		
		KJ_LOG(INFO,prevRound -> participants, unfinished);
		
		KJ_REQUIRE(unfinished.size() > 0, "Internal error");
		
		uint32_t maxSteps = 0;
		
		Round& newRound = prepareRound(unfinished.size());
		prevRound = &(rounds[rounds.size() - 2]);

		auto kDataOut = newRound.kernelData -> getHost().getData();
		for(size_t i = 0; i < unfinished.size(); ++i) {
			// KJ_LOG(INFO,i, unfinished[i], prevRound -> participants.size());
			
			newRound.participants.add(prevRound -> participants[unfinished[i]]);
			
			auto entryOut = kDataOut[i];
			auto entryIn  = kDataIn[unfinished[i]];
			
			entryOut.setState(entryIn.getState());
			entryOut.getState().setEventCount(0);
			
			maxSteps = std::max(maxSteps, entryOut.getState().getNumSteps());
			// KJ_LOG(INFO,entryOut.getState());
		}
		
		if(request.getServiceRequest().getStepLimit() != 0) {
			newRound.kernelRequest -> getHost().getServiceRequest().setStepLimit(
				std::min(request.getServiceRequest().getStepLimit(), maxSteps + ROUND_STEP_LIMIT)
			);
		} else {
			newRound.kernelRequest -> getHost().getServiceRequest().setStepLimit(maxSteps + ROUND_STEP_LIMIT);
		}
		
		KJ_LOG(INFO,"Relaunching", maxSteps, ROUND_STEP_LIMIT);
		
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
			FSC_KARG(mappingData, NOCOPY), FSC_KARG(geoMappingData, NOCOPY)
		);
	}
	
	bool isFinished(FLTKernelData::Entry::Reader entry) {
		auto stopReason = entry.getStopReason();
		
		if(stopReason == FLTStopReason::UNKNOWN) {
			KJ_FAIL_REQUIRE("Kernel stopped for unknown reason");
		}
		
		if(stopReason == FLTStopReason::EVENT_BUFFER_FULL || stopReason == FLTStopReason::COULD_NOT_STEP)
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
			KJ_LOG(INFO,"Trace Finished");
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
	Temporary<FLTConfig> config;
	
	FLTImpl(Own<DeviceBase> device, FLTConfig::Reader config) : device(mv(device)), config(config) {}
	
	Promise<void> trace(TraceContext ctx) override {
		auto request = ctx.getParams();
		
		// Request validation
		if(request.getStepSizeControl().isAdaptive()) {
			KJ_REQUIRE(!request.hasMapping(), "Adaptive step size control can only be used without a mapping");
			
			auto adaptive = request.getStepSizeControl().getAdaptive();
			KJ_REQUIRE(adaptive.getMax() >= adaptive.getMin());
			KJ_REQUIRE(adaptive.getRelativeTolerance() >= 0);
		}
		
		if(request.hasMapping()) {
			KJ_REQUIRE(request.getStepSizeControl().isFixed(), "When using a mapping, a fixed step size is required");
		}
			
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
		
		return kj::startFiber(65536, [this, ctx, request](kj::WaitScope& ws) mutable {
			auto& dataService = getActiveThread().dataService();
			
			auto isNull = [&](capnp::Capability::Client c) {
				return capnp::ClientHook::from(mv(c))->isNull();
			};
			
			auto indexDataRef = request.getGeometry().getData();
			auto geoDataRef = request.getGeometry().getBase();
			auto mappingDataRef = request.getMapping();
			
			DataRef<GeometryMapping::MappingData>::Client geoMappingRef = nullptr;
			if(request.hasGeometryMapping()) {
				mappingDataRef = request.getGeometryMapping().getBase();
				geoMappingRef = request.getGeometryMapping().getData();
			}
			
			KJ_REQUIRE(request.hasField(), "Must specify a magnetic field");
			
			auto downloadField = dataService.download(request.getField().getData()).eagerlyEvaluate(nullptr);
			auto downloadIndexData = dataService.downloadIfNotNull(indexDataRef);
			auto downloadGeometryData =	dataService.downloadIfNotNull(geoDataRef).eagerlyEvaluate(nullptr);
			auto downloadMappingData = dataService.downloadIfNotNull(mappingDataRef).eagerlyEvaluate(nullptr);
			auto downloadGeoMapapping = dataService.downloadIfNotNull(geoMappingRef).eagerlyEvaluate(nullptr);
			
			auto indexData = downloadIndexData.wait(ws);
			auto geometryData = downloadGeometryData.wait(ws);
			auto fieldData = downloadField.wait(ws);
			auto mappingData = downloadMappingData.wait(ws);
			auto geoMappingData = downloadGeoMapapping.wait(ws);
			
			// Extract kernel request
			Temporary<FLTKernelRequest> kernelRequest;
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
			
			TraceCalculation calc(
				*device, mv(kernelRequest), mv(field), mv(positions),
				request.getGeometry(), mv(indexData), geometryData,
				mappingData, geoMappingData,
				config
			);
			calc.run().wait(ws);	
			
			// TODO: This function is WAY TOO LONG. It needs to be broken up into pieces.
			auto postProcessing = [ctx, &calc, request, startPointShape, nStartPoints, geometryData = mv(geometryData)]() mutable {
				auto applyPointShape = [&](auto builder, kj::ArrayPtr<const int64_t> preShape, kj::ArrayPtr<const int64_t> postShape) {
					const size_t shapeSize = startPointShape.size() - 1 + preShape.size() + postShape.size();
					size_t shapeProd = nStartPoints;
					
					auto shape = builder.initShape(shapeSize);
					for(auto i : kj::indices(preShape)) {
						shape.set(i, preShape[i]);
						shapeProd *= preShape[i];
					}
					
					for(auto i : kj::range(1, startPointShape.size())) {
						shape.set(preShape.size() + i - 1, startPointShape[i]);
						// startPointShape is in shapeProd via nStartPoints
					}
					
					for(auto i : kj::indices(postShape)) {
						shape.set(preShape.size() + startPointShape.size() - 1 + i, postShape[i]);
						shapeProd *= postShape[i];
					}
					
					if(builder.hasData()) {
						KJ_REQUIRE(shapeProd == builder.getData().size(), "Internal error, mismatch between shape product and output tensor size", preShape, nStartPoints, postShape);
					} else if(shapeProd > 0) {
						builder.initData(shapeProd);
					}
				};
				
				int64_t nTurns = 0;
				
				auto resultBuilder = kj::heapArrayBuilder<Temporary<FLTKernelData::Entry>>(nStartPoints);
				for(size_t i = 0; i < nStartPoints; ++i)
					resultBuilder.add(calc.consolidateRuns(i));
				
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
				applyPointShape(results.getStopReasons(), {}, {});
				applyPointShape(results.getNumSteps(), {}, {});
				
				KJ_IF_MAYBE(pGeometryData, geometryData) {
					auto geometry = pGeometryData -> get();
					
					auto tagNames = geometry.getTagNames();
					auto outTagNames = results.initTagNames(tagNames.size());
					for(auto i : kj::indices(outTagNames))
						outTagNames.set(i, tagNames[i]);
					
					applyPointShape(results.getEndTags(), {tagNames.size()}, {});
				} else {
					applyPointShape(results.getEndTags(), {0}, {});
				}
				
				auto stopReasonData = results.getStopReasons().getData();
				auto endTagData = results.getEndTags().getData();
				auto stepCountData = results.getNumSteps().getData();
				
				size_t nRecorded = 0;
				
				for(int64_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
					auto entry = kData[iStartPoint].asReader();
					auto state = entry.getState();
					auto events = entry.getEvents();
					
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
								
					int64_t iTurn = 0;
					double lastNewTurnDistance = 0;
					
					for(auto iEvt : kj::indices(events)) {
						auto evt = events[iEvt];
						
						KJ_REQUIRE(!evt.isNotSet(), "Internal error: Event not set", iStartPoint, iEvt);
												
						if(evt.isNewTurn()) {
							iTurn = evt.getNewTurn();
							lastNewTurnDistance = evt.getDistance();
						} else if(evt.isPhiPlaneIntersection()) {
							int64_t iTurnActual = iTurn;
							
							// Due to numerical inaccuracy, it can happen that the event for the new turn
							// and an event for an almost identical phi crossing get emitted in inconsistent
							// order. The rule is: If the two planes are (within tolerance) same, the event should
							// always count from the PREVIOUS turn (so Poincare hits register at the end of the turn,
							// not the beginning).
							
							if(evt.getDistance() - lastNewTurnDistance < 1e-8) {
								if(iTurn == 0) continue;
								iTurnActual = iTurn - 1;
							}
							// It can happen that a phi intersection event lying after
							// the last turn increment is emitted (due to event sorting)
							// These need to be clipped from the result tensor.
							if(iTurnActual >= nTurns)
								continue;
							
							auto ppi = evt.getPhiPlaneIntersection();
							
							auto loc = getEventLocation(evt);
							for(int64_t iDim = 0; iDim < 3; ++iDim) {
								pcCuts(iTurnActual, iStartPoint, ppi.getPlaneNo(), iDim) = loc[iDim];
							}
							pcCuts(iTurnActual, iStartPoint, ppi.getPlaneNo(), 3) = forwardLCs[iEvt];
							pcCuts(iTurnActual, iStartPoint, ppi.getPlaneNo(), 4) = backwardLCs[iEvt];
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
					
					stepCountData.set(iStartPoint, state.getNumSteps());
				}
				writeTensor(pcCuts, results.getPoincareHits());
				applyPointShape(results.getPoincareHits(), {5, nSurfs}, {nTurns});
				
				writeTensor(endPoints, results.getEndPoints());
				applyPointShape(results.getEndPoints(), {4}, {});
				
				if(nRecorded > 0) {
					KJ_REQUIRE(
						3 * nRecorded * nStartPoints * capnp::ELEMENTS <= capnp::MAX_LIST_ELEMENTS,
						"The number of recorded points is so large that the returned"
						" tensor would exceed size limitations. Please reduce the number"
						" of points, increase step size, or increase no. of steps between"
						" recordings"
					);
					
					Tensor<double, 3> fieldLines(nRecorded, nStartPoints, 3);
					fieldLines.setConstant(std::nan(""));
					
					Tensor<double, 2> fieldStrengths(nRecorded, nStartPoints);
					fieldStrengths.setConstant(std::nan(""));
					
					for(auto iStartPoint : kj::range(0, nStartPoints)) {
						auto entry = kData[iStartPoint].asReader();
						auto events = entry.getEvents();
						
						int64_t iRecord = 0;
						for(auto iEvt : kj::indices(events)) {
							auto evt = events[iEvt];
							
							// Process only "record" events
							if(!evt.isRecord())
								continue;
							
							auto loc = getEventLocation(evt);
							for(int32_t iDim = 0; iDim < 3; ++iDim)
								fieldLines(iRecord, iStartPoint, iDim) = loc[iDim];
							
							fieldStrengths(iRecord, iStartPoint) = evt.getRecord().getFieldStrength();
							
							++iRecord;
						}
					}
					
					writeTensor(fieldLines, results.getFieldLines());
					writeTensor(fieldStrengths, results.getFieldStrengths());
				}
				
				applyPointShape(results.getFieldLines(), {3}, {(int64_t) nRecorded});
				applyPointShape(results.getFieldStrengths(), {}, {(int64_t) nRecorded});
				
				if(request.getFieldLineAnalysis().isCalculateIota()) {
					auto iotas = results.getFieldLineAnalysis().initIotas();
					applyPointShape(iotas, {}, {});
					auto iotaData = iotas.getData();
					
					for(auto iStartPoint : kj::indices(iotaData)) {
						auto state = kData[iStartPoint].getState();
						
						double myIota = state.getTheta() / state.getPhi();
						iotaData.set(iStartPoint, myIota);
					}
				}
				
				if(request.getFieldLineAnalysis().isCalculateFourierModes()) {
					using FP = nudft::FourierPoint<2, 2>;
					using FM = nudft::FourierMode<2, 2>;
					auto calcFM = request.getFieldLineAnalysis().getCalculateFourierModes();
					KJ_REQUIRE(calcFM.getIota().getData().size() == nStartPoints, "Incorrect iota count specified");
					
					// We use expansions of the form sin(m * theta - n * phi)
					// Therefore phiMultiplier must be negative
					const double phiMultiplier = -static_cast<double>(calcFM.getToroidalSymmetry()) / calcFM.getIslandM();					
					
					const int maxM = calcFM.getMMax();
					const int maxN = calcFM.getNMax();
					
					const int64_t nToroidalCoeffs = 2 * maxN + 1;
					const int64_t nPoloicalCoeffs = maxM + 1;
					
					const double aliasThreshold = calcFM.getModeAliasingThreshold();
					
					auto toroidalIndex = [&](int n) -> int64_t {
						if(n >= 0) return n;
						return nToroidalCoeffs + n;
					};
					
					Tensor<double, 3> rCos(nPoloicalCoeffs, nToroidalCoeffs, nStartPoints);
					Tensor<double, 3> rSin(nPoloicalCoeffs, nToroidalCoeffs, nStartPoints);
					Tensor<double, 3> zCos(nPoloicalCoeffs, nToroidalCoeffs, nStartPoints);
					Tensor<double, 3> zSin(nPoloicalCoeffs, nToroidalCoeffs, nStartPoints);
					
					Tensor<double, 2> nTor(nPoloicalCoeffs, nToroidalCoeffs);
					Tensor<double, 2> mPol(nPoloicalCoeffs, nToroidalCoeffs);
					
					rCos.setConstant(0);
					rSin.setConstant(0);
					zCos.setConstant(0);
					zSin.setConstant(0);
					nTor.setConstant(0);
					mPol.setConstant(0);
					
					Tensor<double, 1> t0(nStartPoints);
					
					// Index of n = 0, m = 1 mode
					const size_t n0m1Index = 1;
					
					// #pragma omp parallel for
					for(int64_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
						auto entry = kData[iStartPoint].asReader();
						auto state = entry.getState();
						auto events = entry.getEvents();
						
						const double iota = calcFM.getIota().getData()[iStartPoint];
					
						// Prepare the modes we want to analyze
						kj::Vector<FM> modes;
						
						//for(auto n : kj::range(-maxN, maxN + 1)) {
						for(auto in : kj::range(0, nToroidalCoeffs)) {
							auto n = in <= maxN ? in : in - nToroidalCoeffs;
							
							for(auto m : kj::range(0, maxM + 1)) {
								nTor(m, toroidalIndex(n)) = n * phiMultiplier;
								mPol(m, toroidalIndex(n)) = m;
								
								double parallelAngle = std::abs(n * phiMultiplier + m * iota);
								
								// bool isResonant = std::abs(n * phiMultiplier * iota - m) < aliasThreshold;
								
								if(m == 0 && n < 0) {
									continue;
								}
								
								kj::Maybe<FM&> modeAliases = nullptr;
								for(auto& prevMode : modes) {
									double prevParAngle = std::abs(prevMode.coeffs[0] * phiMultiplier + prevMode.coeffs[1] * iota);
									
									if(std::abs(parallelAngle - prevParAngle) < aliasThreshold) {
										// KJ_DBG("Mode alias", n, m, parallelAngle, prevMode.coeffs[0], prevMode.coeffs[1], prevParAngle);
										modeAliases = prevMode;
									}
								}
								
								KJ_IF_MAYBE(pOther, modeAliases) {
									// double myResonance = std::abs(n * phiMultiplier * iota - m);
									// double otherResonance = std::abs(pOther -> coeffs[0] * phiMultiplier * iota - pOther -> coeffs[1]);
									
									// We give priority to the lowest-m mode (axis variations tend to be the biggest source of non-stationary
									// variations, unless the other mode is the 0/1 mode, which is expected to be huge
																		
									const int on = pOther -> coeffs[0];
									const int om = pOther -> coeffs[1];
									
									bool keepMe = m < om;
									bool keepOther = !keepMe;
									
									if((m == 0 && n == 0) || (m == 1 && n == 0)) keepMe = true;
									if((om == 0 && on == 0) || (om == 1 && on == 0)) keepOther = true;
									
									// KJ_DBG(keepMe, keepOther);
									
									if(keepMe && !keepOther) {
										pOther -> coeffs[0] = n;
										pOther -> coeffs[1] = m;
										continue;
									} else if(keepOther && !keepMe) {
										continue;
									}
								}
								
								FM mode;
								mode.coeffs[0] = n;
								mode.coeffs[1] = m;
								modes.add(mode);
							}
						}
							
						kj::Vector<FP> fourierPoints;
						for(auto evt : events) {
							if(!evt.isFourierPoint())
								continue;
							
							auto evtPoint = evt.getFourierPoint();
							double x = evt.getX();
							double y = evt.getY();
							double z = evt.getZ();
							double r = std::sqrt(x*x + y*y);
							
							FP point;
							point.angles[0] = evtPoint.getPhi() * phiMultiplier;
							point.angles[1] = evtPoint.getPhi() * iota;
							point.y[0] = r;
							point.y[1] = z;
							fourierPoints.add(point);
						}
						
						nudft::calculateModes<2, 2>(fourierPoints.asPtr(), modes);
						double theta0 = std::atan2(modes[n0m1Index].sinCoeffs[0], modes[n0m1Index].cosCoeffs[0]);
						t0(iStartPoint) = theta0;
						
						// Subtract theta0 from poloidal angle
						for(auto& p : fourierPoints) {
							p.angles[1] -= theta0;
						}
						
						// Now do the full Fourier calculation						
						nudft::calculateModes<2, 2>(fourierPoints.asPtr(), modes);
						
						for(auto mode : modes) {
							rCos(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), iStartPoint) = mode.cosCoeffs[0];
							zCos(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), iStartPoint) = mode.cosCoeffs[1];
							rSin(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), iStartPoint) = mode.sinCoeffs[0];
							zSin(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), iStartPoint) = mode.sinCoeffs[1];
						}
					}
					
					auto out = results.getFieldLineAnalysis().initFourierModes();
					
					auto surf = out.initSurfaces();
					surf.setNTor(maxN);
					surf.setMPol(maxM);
					surf.setToroidalSymmetry(calcFM.getToroidalSymmetry());
					surf.setNTurns(calcFM.getIslandM());
					writeTensor(rCos, surf.getRCos());
					writeTensor(zSin, surf.getZSin());
					writeTensor(t0, out.getTheta0());
					
					applyPointShape(surf.getRCos(), {}, {nToroidalCoeffs, nPoloicalCoeffs});
					applyPointShape(surf.getZSin(), {}, {nToroidalCoeffs, nPoloicalCoeffs});
					applyPointShape(out.getTheta0(), {}, {});
					
					if(!calcFM.getStellaratorSymmetric()) {
						auto ns = surf.initNonSymmetric();
						writeTensor(rSin, ns.getRSin());
						writeTensor(zCos, ns.getZCos());
						applyPointShape(ns.getRSin(), {}, {nToroidalCoeffs, nPoloicalCoeffs});
						applyPointShape(ns.getZCos(), {}, {nToroidalCoeffs, nPoloicalCoeffs});
					}
					
					auto oNTor = out.initNTor(2 * maxN + 1);
					for(int i : kj::indices(oNTor)) {
						oNTor.set(i, -nTor(0, i));
					}
					
					auto oMPol = out.initMPol(maxM + 1);
					for(int i : kj::indices(oMPol))
						oMPol.set(i, mPol(i, 0));
				}
			};
			
			getActiveThread().worker().executeAsync(mv(postProcessing)).wait(ws);
		}).attach(thisCap());
	}
	
	Promise<void> findAxis(FindAxisContext ctx) override {
		auto req = thisCap().findAxisBatchRequest();
		
		auto points = req.initPoints();
		points.setShape({3});
		points.setData(ctx.getParams().getStartPoint());
		
		req.setRequest(ctx.getParams());
		
		return req.send()
		.then([ctx](auto response) mutable {
			auto results = ctx.initResults();
			results.setPos(response.getPos().getData());
			results.setAxis(response.getAxis());
			results.setMeanField(response.getMeanField().getData()[0]);
		});
	}
		
	Promise<void> findAxisBatch(FindAxisBatchContext ctx) override {
		auto points = heapHeld<Tensor<double, 2>>();
		auto vardimShape = readVardimTensor(ctx.getParams().getPoints(), 1, *points);
		size_t nPoints = points -> dimension(0);
		
		struct IterResult {
			double r;
			double z;
		};
		
		auto params = ctx.getParams().getRequest();
		auto nPhi = params.getNPhi();
		auto islandM = params.getIslandM();
		
		auto axis = heapHeld<Tensor<double, 3>>(nPhi * islandM, nPoints, 3);
		auto fields = heapHeld<Tensor<double, 1>>(nPoints);
		auto pos = heapHeld<Tensor<double, 2>>(nPoints, 3);
		
		{
			constexpr double nan = std::numeric_limits<double>::quiet_NaN();
			axis -> setConstant(nan);
			fields -> setConstant(nan);
			pos -> setConstant(nan);
		}
		
		Temporary<ComputedField> field =  params.getField();
		field.setData(
			getActiveThread().dataService().download(field.getData())
			.then([](auto ref) -> DataRef<Float64Tensor>::Client { return ref; })
		);
						
		auto jobs = kj::heapArrayBuilder<Promise<void>>(nPoints);
		for(auto iPoint : kj::range(0, nPoints)) {
			double x = (*points)(iPoint, 0);
			double y = (*points)(iPoint, 1);
			double z = (*points)(iPoint, 2);
			
			double phi = atan2(y, x);
			
			auto performIteration = [this, params, phi, field = field.asReader()](IterResult iter) mutable {
				double r = iter.r;
				double z = iter.z;
				
				KJ_LOG(INFO,"FLT::findAxis iteration", r, z);
				
				bool isNan = false;
				if(r != r) isNan = true;
				if(z != z) isNan = true;
				
				KJ_REQUIRE(!isNan, "The magnetic axis search failed to converge. This likely means the starting point lies outside the magnetic surface domains. Please specify a different starting point.");
				
				// Trace from starting position
				auto req = thisCap().traceRequest();
				auto sp = req.initStartPoints();
				
				sp.setShape({3});
				sp.setData({r * cos(phi), r * sin(phi), z});
				req.setTurnLimit(params.getNTurns());
				req.setStepSize(params.getStepSize());
				req.setField(field);
				
				if(params.hasMapping())
					req.setMapping(params.getMapping());
				
				if(params.hasGeometryMapping())
					req.setGeometryMapping(params.getGeometryMapping());
				
				if(params.getStepSizeControl().isAdaptive())
					req.getStepSizeControl().setAdaptive(params.getStepSizeControl().getAdaptive());
				
				size_t nSym = params.getField().getGrid().getNSym();
				auto islandM = params.getIslandM();
				auto planes = req.initPlanes(nSym);
				for(auto i : kj::indices(planes)) {
					planes[i].getOrientation().setPhi(phi + (i + 1) * 2 * pi / nSym);
				}
				
				return req.send()
				.then([phi, nSym, islandM](capnp::Response<FLTResponse> response) mutable {
					// After tracing, take mean of points
					Tensor<double, 3> result;
					readTensor(response.getPoincareHits(), result);
					KJ_REQUIRE(result.dimension(1) == nSym);
					KJ_REQUIRE(result.dimension(2) == 5);
					
					uint32_t n = 0;
					double r = 0;
					double z = 0;
					
					size_t iPoint = 0;
					for(auto iTurn : kj::range(0, result.dimension(0))) {
						for(auto iPlane : kj::range(0, result.dimension(1))) {
							if(++iPoint % islandM == 0) {
								double x = result(iTurn, iPlane, 0);
								double y = result(iTurn, iPlane, 1);
								
								double rContrib = sqrt(x * x + y * y);
								double zContrib = result(iTurn, iPlane, 2);
								
								if(rContrib == rContrib) {								
									r += rContrib;
									z += zContrib;
									++n;
								}
							}
						}
					}
					
					return IterResult { r / n, z / n };
				});
			};
			
			auto traceAxis = [this, params, phi, iPoint, field = field.asReader(), &axis = *axis, &fields = *fields, &pos = *pos](IterResult iter) mutable {
				double r = iter.r;
				double z = iter.z;
				
				auto islandM = params.getIslandM();
				
				auto req = thisCap().traceRequest();
				auto sp = req.initStartPoints();
				sp.setShape({3});
				sp.setData({r * cos(phi), r * sin(phi), z});
				
				// Copy position into output
				pos(iPoint, 0) = r * cos(phi);
				pos(iPoint, 1) = r * sin(phi);
				pos(iPoint, 2) = z;
				
				req.setTurnLimit(islandM);
				req.setStepSize(params.getStepSize());
				req.setField(field);
				
				if(params.hasMapping())
					req.setMapping(params.getMapping());
				
				if(params.getStepSizeControl().isAdaptive())
					req.getStepSizeControl().setAdaptive(params.getStepSizeControl().getAdaptive());
				
				// req.setRecordEvery(1);
				auto nPhi = params.getNPhi();
				auto planes = req.initPlanes(nPhi);
				for(auto i : kj::indices(planes))
					planes[i].getOrientation().setPhi(2 * i * pi / nPhi);
				
				return req.send()
				.then([nPhi, islandM, phi, &axis, &fields, iPoint](capnp::Response<FLTResponse> response) mutable {
					// Extract field line
					Tensor<double, 3> pcHits;
					readTensor(response.getPoincareHits(), pcHits);
					KJ_REQUIRE(pcHits.dimension(0) == islandM);    // nTurns
					KJ_REQUIRE(pcHits.dimension(1) == nPhi); // nPlanes
					KJ_REQUIRE(pcHits.dimension(2) == 5);    // x, y, z, lc_fwd, lc_bwd
					
					double phiStartPositive = fmod(phi + 2 * pi, 2 * pi);
					
					for(auto iTurn : kj::range(0, islandM)) {
						for(auto iPlane : kj::range(0, nPhi)) {
							double planePhi = 2 * iPlane * pi / nPhi;
							
							// Planes that are before phiStart need to be extracted from
							// the previous turn. This includes the starting plane.
							size_t iTurnActual = planePhi > phiStartPositive ? iTurn : iTurn + (islandM - 1);
							iTurnActual %= islandM;
							
							axis(iPlane + iTurn * nPhi, iPoint, 0) = pcHits(iTurnActual, iPlane, 0);
							axis(iPlane + iTurn * nPhi, iPoint, 1) = pcHits(iTurnActual, iPlane, 1);
							axis(iPlane + iTurn * nPhi, iPoint, 2) = pcHits(iTurnActual, iPlane, 2);
						}
					}
															
					// Compute mean field along axis
					double accum = 0;
					for(double fs : response.getFieldStrengths().getData()) {
						accum += fs;
					}
					fields(iPoint) = accum / response.getFieldStrengths().getData().size();
				});
			};
						
			double r = sqrt(x * x + y * y);
			
			Promise<IterResult> nIterResult = IterResult { r, z };
			
			for(auto iteration : kj::range(0, params.getNIterations())) {
				nIterResult = nIterResult.then(cp(performIteration));
			}
			
			jobs.add(nIterResult.then(mv(traceAxis)).catch_([iPoint](kj::Exception&& e) {}));
		}
		
		return kj::joinPromisesFailFast(jobs.finish())
		.then([ctx, axis, fields, pos, vs = mv(vardimShape)]() mutable {
			auto res = ctx.initResults();
			
			writeVardimTensor(*axis, 1, vs, res.getAxis());
			writeVardimTensor(*pos, 1, vs, res.getPos());
			writeVardimTensor(*fields, 0, vs, res.getMeanField());
		})
		.attach(axis.x(), pos.x(), fields.x(), points.x(), mv(field));
	}
	
	Promise<void> findLcfs(FindLcfsContext ctx) override {
		auto params = ctx.getParams();
		
		struct Process {
			FindLcfsContext ctx;
			FLT::Client flt;
			
			Vec3d x1;
			Vec3d x2;
			
			Process(FindLcfsContext ctx, FLT::Client _flt) :
				ctx(ctx), flt(mv(_flt))
			{
				auto params = ctx.getParams();
				
				auto p1 = params.getP1();
				auto p2 = params.getP2();
				
				KJ_REQUIRE(p1.size() == 3);
				KJ_REQUIRE(p2.size() == 3);
				
				x1 = Vec3d(p1[0], p1[1], p1[2]);
				x2 = Vec3d(p2[0], p2[1], p2[2]);
			}
			
			Promise<void> run() {
				auto params = ctx.getParams();
				
				double dist = (x2 - x1).norm();
				if(dist <= params.getTolerance()) {
					auto res = ctx.initResults();
					auto pos = res.initPos(3);
					for(unsigned int i = 0; i < 3; ++i)
						pos.set(i, x2[i]);
					
					return READY_NOW;
				}
				
				auto nScan = params.getNScan();
				KJ_REQUIRE(nScan >= 2);
				
				auto request = flt.traceRequest();
				request.setDistanceLimit(params.getDistanceLimit());
				request.setStepSize(params.getStepSize());
				request.setCollisionLimit(1);
				request.setField(params.getField());
				request.setGeometry(params.getGeometry());
				
				if(params.hasMapping())
					request.setMapping(params.getMapping());
				
				if(params.hasGeometryMapping())
					request.setGeometryMapping(params.getGeometryMapping());
			
				if(params.getStepSizeControl().isAdaptive())
					request.getStepSizeControl().setAdaptive(params.getStepSizeControl().getAdaptive());
				
				Tensor<double, 2> points(nScan, 3);
				Vec3d dx = (x2 - x1) / (nScan + 1);
				for(auto iScan : kj::range(0, nScan)) {
					Vec3d xNew = x1 + dx * (iScan + 1);
					for(unsigned int iDim = 0; iDim < 3; ++iDim)
						points(iScan, iDim) = xNew(iDim);
				}
				
				writeTensor(points, request.getStartPoints());
				
				return request.send()
				.then([this, nScan, dx](auto response) {
					auto stopReasons = response.getStopReasons().getData();
					
					uint32_t firstClosed = nScan;
					for(auto i : kj::range(0, nScan)) {
						if(stopReasons[i] == FLTStopReason::DISTANCE_LIMIT) {
							firstClosed = i;
							break;
						}
					}
					
					x2 = x1 + (firstClosed + 1) * dx;
					x1 = x1 + firstClosed * dx;
					
					return run();
				});
			}
		};
		
		auto proc = kj::heap<Process>(ctx, thisCap());
		auto result = proc -> run();
		return result.attach(mv(proc));
	}
};
	
}

namespace fsc {	
	// TODO: Make this accept data service instead	
	Own<FLT::Server> newFLT(Own<DeviceBase> device, FLTConfig::Reader config) {
		return kj::heap<FLTImpl>(mv(device), config);
	}
}
