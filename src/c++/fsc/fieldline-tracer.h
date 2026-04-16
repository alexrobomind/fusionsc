#pragma once

#include "flt-kernels.h"

#include "kernels/launch.h"
#include "kernels/message.h"
#include "kernels/tensor.h"
#include "kernels/karg.h"

#include "tensor.h"
#include "random-kernels.h"
#include "nudft.h"

#include <array>
#include <algorithm>
#include <limits>

#include <kj/vector.h>
#include <fsc/flt.capnp.cu.h>

namespace fsc {

// Constants for fieldline tracing
constexpr size_t FLT_STEPS_PER_ROUND = 1000;
constexpr size_t FLT_EVENTBUF_SIZE = 2500;
constexpr size_t FLT_EVENTBUF_SIZE_NOGEO = 100;
constexpr uint32_t FLT_ROUND_STEP_LIMIT = 1000000;

/**
 * \ingroup kernels
 * \brief Manages the execution of fieldline tracing rounds
 * 
 * This class handles:
 * - Device and data management for fieldline tracing
 * - Round preparation and initialization
 * - Event buffer size calculation
 * - Kernel launching and result consolidation
 * 
 * Designed to be a self-contained tracing engine that can be used
 * independently of the FLT service interface.
 */
class FieldlineTracer {
public:
    /**
     * \brief Construct a new FieldlineTracer
     * 
     * \param device The device to run traces on
     * \param request The trace request configuration
     * \param field The magnetic field data (4D tensor)
     * \param positions Initial positions (3 x nPositions tensor)
     * \param geometryIndex Geometry index reader
     * \param indexData Optional indexed geometry index data
     * \param geometryData Optional merged geometry data
     * \param mappingData Optional reversible fieldline mapping data
     * \param geoMappingData Optional geometry mapping data
     * \param config FLT configuration
     */
    FieldlineTracer(
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
    );

    /**
     * \brief Run the fieldline tracing
     * 
     * Returns a promise that resolves when all traces are complete.
     * Uses recursive round execution until all participants finish.
     */
    Promise<void> run();

    /**
     * \brief Consolidate results for a single participant across all rounds
     * 
     * \param participantIndex Index of the participant to consolidate
     * \return Temporary entry containing consolidated events and state
     */
    Temporary<FLTKernelData::Entry> consolidateRuns(size_t participantIndex);

    /**
     * \brief Get the number of rounds executed
     */
    size_t getRoundCount() const { return rounds.size(); }
    
    /**
     * \brief Get consolidated data for all participants
     * 
     * Convenience method to get consolidated runs for all participants.
     * Returns a vector of Temporary entries.
     */
    kj::Vector<Temporary<FLTKernelData::Entry>> getConsolidatedData();
    
private:
    struct Round {
        FSC_BUILDER_MAPPING(::fsc, FLTKernelData) kernelData;
        FSC_BUILDER_MAPPING(::fsc, FLTKernelRequest) kernelRequest;
        
        kj::Vector<size_t> participants;
        size_t upperBound;
    };

    // Device and data
    Own<DeviceBase> device;
    Tensor<double, 2> positions;
    DeviceMappingType<Own<TensorMap<const Tensor<double, 4>>>> field;
    
    // Request and rounds
    Temporary<FLTKernelRequest> request;
    kj::Vector<Round> rounds;
    
    // Geometry and mapping data
    FSC_BUILDER_MAPPING(::fsc, IndexedGeometry) indexedGeometry;
    FSC_READER_MAPPING(::fsc, IndexedGeometry::IndexData) indexData;
    FSC_READER_MAPPING(::fsc, MergedGeometry) geometryData;
    FSC_READER_MAPPING(::fsc, ReversibleFieldlineMapping) mappingData;
    FSC_READER_MAPPING(::fsc, GeometryMapping::MappingData) geoMappingData;
    
    // Configuration
    Temporary<FLTConfig> config;
    uint32_t eventBufferSize;
    
    // Helper methods
    uint32_t computeBufferSize(FLTConfig::EventBuffer::Reader config, uint32_t nPoints);
    Round& prepareRound(size_t nParticipants);
    Round& setupInitialRound();
    Round& setupFollowupRound();
    Round& setupRound();
    
    Promise<void> startRound(Round& r);
    bool isFinished(FLTKernelData::Entry::Reader entry);
    bool isFinished(Round& r);
    Promise<void> runRound();
};

} // namespace fsc
