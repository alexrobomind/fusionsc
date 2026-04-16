#pragma once

#include <fsc/flt.capnp.cu.h>

#include "tensor.h"
#include "nudft.h"
#include "interpolation.h"
#include "geometry.h"

#include <vector>
#include <limits>

namespace fsc {

/**
 * \ingroup kernels
 * \brief Post-processing functions for fieldline tracing results
 * 
 * This module handles all post-processing of trace results, including:
 * - Poincaré section extraction and formatting
 * - End point and normal calculation
 * - Field line extraction
 * - Fourier mode calculation
 * - Iota calculation
 * - Shape manipulation for output tensors
 */
namespace fieldline_postprocessing {

/**
 * \brief Apply point shape transformation to tensor builder
 * 
 * Handles the transformation of tensor shapes by adding pre- and post-shape
 * dimensions to the main point shape.
 * 
 * \param builder The capnp builder to modify
 * \param preShape Pre-shape dimensions to prepend
 * \param postShape Post-shape dimensions to append
 * \param startPointShape Original shape (minus the first dimension which is coordinates)
 * \param nStartPoints Total number of start points
 */
template<typename BuilderType>
void applyPointShape(
    BuilderType& builder,
    kj::ArrayPtr<const int64_t> preShape,
    kj::ArrayPtr<const int64_t> postShape,
    kj::ArrayPtr<const int64_t> startPointShape,
    int64_t nStartPoints
);

/**
 * \brief Process a single trace entry and extract results
 * 
 * Consolidates events for a single participant and extracts:
 * - Poincaré section hits
 * - End points
 * - Normals at collision points
 * - Stop reasons
 * - Tag information
 * 
 * \param entry The trace entry to process
 * \param events The events from this entry
 * \param geometryData Optional geometry data for tag extraction
 * \param forwardLCs Forward free path lengths
 * \param backwardLCs Backward free path lengths
 * \param state The final state
 * \param resultBuilder Output builder for results
 * \param startPointShape Original shape
 * \param nSurfs Number of Poincaré planes
 * \param nTurns Maximum number of turns
 * \param recordMode How to record plane intersections
 */
void processTraceEntry(
    FLTKernelData::Entry::Reader entry,
    kj::ArrayPtr<FLTKernelEvent::Reader> events,
    Maybe<MergedGeometry::Reader> geometryData,
    kj::ArrayPtr<const double> forwardLCs,
    kj::ArrayPtr<const double> backwardLCs,
    FLTKernelData::Entry::State::Reader state,
    capnp::StructBuilder& resultBuilder,
    kj::ArrayPtr<const int64_t> startPointShape,
    int64_t nSurfs,
    int64_t nTurns,
    FLTRequest::PlaneIntersectionRecordMode recordMode,
    const FLTRequest& request
);

/**
 * \brief Extract and format Poincaré section data
 * 
 * Processes events to extract plane intersections and formats them
 * according to the record mode (every hit or last per turn).
 * 
 * \param events Events from the trace
 * \param pcCuts Output tensor for Poincaré cuts
 * \param nSurfs Number of Poincaré planes
 * \param nTurns Number of turns
 * \param recordMode How to record intersections
 * \param request The original request
 */
void extractPoincareHits(
    kj::ArrayPtr<FLTKernelEvent::Reader> events,
    Tensor<double, 4>& pcCuts,
    int64_t nSurfs,
    int64_t nTurns,
    FLTRequest::PlaneIntersectionRecordMode recordMode,
    const FLTRequest& request
);

/**
 * \brief Extract end points and normals from trace
 * 
 * Computes end points and surface normals at collision points.
 * 
 * \param state Final state
 * \param stopReason Stop reason from entry
 * \param lastCollision Optional last collision event
 * \param geometryData Geometry data for normal calculation
 * \param geoMappingData Geometry mapping data
 * \param mappingData Mapping data
 * \param endPoints Output tensor for end points
 * \param normals Output tensor for normals
 * \param endTagData Output for end tags
 * \param stopReasonData Output for stop reasons
 */
void extractEndPointsAndNormals(
    FLTKernelData::Entry::State::Reader state,
    FLTStopReason stopReason,
    Maybe<FLTKernelEvent::Reader> lastCollision,
    Maybe<MergedGeometry::Reader> geometryData,
    Maybe<GeometryMapping::MappingData::Reader> geoMappingData,
    ReversibleFieldlineMapping::Reader mappingData,
    Tensor<double, 2>& endPoints,
    Tensor<double, 2>& normals,
    Float64Tensor::Builder endTagData,
    Float64Tensor::Builder stopReasonData
);

/**
 * \brief Extract recorded field lines and strengths
 * 
 * Processes "record" events to extract field line positions
 * and field strength values.
 * 
 * \param events Events from the trace
 * \param nRecorded Number of recorded points
 * \param resultBuilder Result builder
 */
void extractFieldLines(
    kj::ArrayPtr<FLTKernelEvent::Reader> events,
    int64_t nRecorded,
    capnp::StructBuilder& resultBuilder
);

/**
 * \brief Calculate iota from trace state
 * 
 * Computes the rotational transform (iota) from the final state.
 * 
 * \param state Final state of the trace
 * \param iotas Output tensor for iota values
 */
void calculateIota(
    FLTKernelData::Entry::State::Reader state,
    Float64Tensor::Builder iotas
);

/**
 * \brief Calculate Fourier modes from trace data
 * 
 * Computes Fourier modes (rCos, rSin, zCos, zSin) from phi and position data.
 * Supports stellarator-symmetric and non-symmetric calculations.
 * 
 * \param events Events containing Fourier point data
 * \param iota Iota value for this trace
 * \param calcFM Fourier mode calculation parameters
 * \param resultBuilder Result builder
 */
void calculateFourierModes(
    kj::ArrayPtr<FLTKernelEvent::Reader> events,
    double iota,
    FLTRequest::FieldLineAnalysis::CalculateFourierModes::Reader calcFM,
    capnp::StructBuilder& resultBuilder
);

/**
 * \brief Get event location as 3D vector
 */
std::array<double, 3> getEventLocation(FLTKernelEvent::Reader e);

/**
 * \brief Initialize result tensors with NaN
 * 
 * \param pcCuts Poincaré cuts tensor
 * \param endPoints End points tensor
 * \param normals Normals tensor
 */
void initializeResultTensors(
    Tensor<double, 4>& pcCuts,
    Tensor<double, 2>& endPoints,
    Tensor<double, 2>& normals
);

} // namespace fieldline_postprocessing

} // namespace fsc
