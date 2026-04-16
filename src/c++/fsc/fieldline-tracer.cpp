#include "fieldline-tracer.h"

#include <algorithm>

namespace fsc {
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

} // anonymous namespace

FieldlineTracer::FieldlineTracer(
    Own<DeviceBase>& device,
    Temporary<FLTKernelRequest>& request,
    Own<TensorMap<const Tensor<double, 4>>>& field,
    Tensor<double, 2>& positions,
    IndexedGeometry::Reader geometryIndex,
    Maybe<LocalDataRef<IndexedGeometry::IndexData>> indexData,
    Maybe<LocalDataRef<MergedGeometry>> geometryData,
    Maybe<LocalDataRef<ReversibleFieldlineMapping>> mappingData,
    Maybe<LocalDataRef<GeometryMapping::MappingData>> geoMappingData,
    FLTConfig::Reader config
) :
    device(device->addRef()),
    positions(mv(positions)),
    
    field(mapToDevice(mv(field), device, true)),
    
    request(mv(request)),
    
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
    
    // Adjust step limit to a reasonable default if users forget to specify it
    if(request.getServiceRequest().getStepLimit() == 0) {
        request.getServiceRequest().setStepLimit(config.getDefaultStepLimit());
    }
    
    // Hard clamp the step limit
    if(config.getMaxStepLimit() != 0 && request.getServiceRequest().getStepLimit() > config.getMaxStepLimit()) {
        request.getServiceRequest().setStepLimit(config.getMaxStepLimit());
    }
    
    KJ_REQUIRE(request.getServiceRequest().getForwardDirection().which() < 2, "Unknown forward direction type specified. This likely means that this library version is too old to understand your request");
    
    eventBufferSize = computeBufferSize(config.getEventBuffer(), positions.dimension(1));
}

uint32_t FieldlineTracer::computeBufferSize(FLTConfig::EventBuffer::Reader config, uint32_t nPoints) {
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

Round& FieldlineTracer::prepareRound(size_t nParticipants) {
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

Round& FieldlineTracer::setupInitialRound() {
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
        request.getServiceRequest().getStepLimit() != 0 ? std::min(request.getServiceRequest().getStepLimit(), FLT_ROUND_STEP_LIMIT) : FLT_ROUND_STEP_LIMIT
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

Round& FieldlineTracer::setupFollowupRound() {
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
            KJ_LOG(INFO, "Failed to step", i);
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
    
    // Note: This assumption is actually incorrect
    // If all participants had COULD_NOT_STEP and the buffer size can not increase,
    // the round size can get modified to 0
    // KJ_REQUIRE(unfinished.size() > 0, "Internal error");
    
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
            std::min(request.getServiceRequest().getStepLimit(), maxSteps + FLT_ROUND_STEP_LIMIT)
        );
    } else {
        newRound.kernelRequest -> getHost().getServiceRequest().setStepLimit(maxSteps + FLT_ROUND_STEP_LIMIT);
    }
    
    KJ_LOG(INFO,"Relaunching", maxSteps, FLT_ROUND_STEP_LIMIT);
    
    return newRound;
}

Round& FieldlineTracer::setupRound() {
    if(rounds.size() == 0)
        return setupInitialRound();
    else
        return setupFollowupRound();
}
    

Promise<void> FieldlineTracer::startRound(Round& r) {
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

bool FieldlineTracer::isFinished(FLTKernelData::Entry::Reader entry) {
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

bool FieldlineTracer::isFinished(Round& r) {
    for(auto kDatum : r.kernelData -> getHost().getData()) {
        if(!isFinished(kDatum))
            return false;
    }
    
    return true;
}

Promise<void> FieldlineTracer::runRound() {
    auto& round = setupRound();
    return startRound(round);
}
        
Promise<void> FieldlineTracer::run() {
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

Temporary<FLTKernelData::Entry> FieldlineTracer::consolidateRuns(size_t participantIndex) {
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
    
    KJ_REQUIRE(isFinished(result.asReader()) || result.getStopReason() == FLTStopReason::COULD_NOT_STEP);
    
    return result;
}

kj::Vector<Temporary<FLTKernelData::Entry>> FieldlineTracer::getConsolidatedData() {
    kj::Vector<Temporary<FLTKernelData::Entry>> result;
    for(size_t i = 0; i < rounds.empty() ? 0 : rounds[0].participants.size(); ++i) {
        result.add(consolidateRuns(i));
    }
    return result;
}

} // namespace fsc
