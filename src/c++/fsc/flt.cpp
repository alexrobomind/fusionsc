#include "flt.h"

#include "fieldline-tracer.h"
#include "fieldline-postprocessing.h"

#include "kernels/launch.h"
#include "kernels/message.h"
#include "kernels/tensor.h"
#include "kernels/karg.h"

#include "tensor.h"
#include "random-kernels.h"
#include "nudft.h"

#include <algorithm>
#include <limits>

#include <kj/vector.h>
#include <fsc/flt.capnp.cu.h>

using namespace fsc;

namespace {

// Helper function for getting event location
std::array<double, 3> getEventLocation(FLTKernelEvent::Reader e) {
    return {e.getX(), e.getY(), e.getZ()};
}

} // anonymous namespace

namespace fsc {

namespace {

struct TraceContext {
    Own<DeviceBase> device;
    Temporary<FLTConfig> config;
    FLT::Client flt;
    
    TraceContext(Own<DeviceBase> device, FLTConfig::Reader config, FLT::Client flt)
        : device(mv(device)), config(config), flt(mv(flt)) {}
};

} // anonymous namespace

struct FLTImpl : public FLT::Server {
    Own<DeviceBase> device;
    Temporary<FLTConfig> config;
    
    FLTImpl(Own<DeviceBase> device, FLTConfig::Reader config) : device(mv(device)), config(config) {}
    
    Promise<void> trace(TraceContext ctx) override {
        auto request = ctx.getParams();
        
        // Request validation
        if(request.getStepSizeControl().isAdaptive()) {
            KJ_REQUIRE(!request.hasMapping() && !request.hasGeometryMapping(), "Adaptive step size control can only be used without a mapping");
            
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
            auto downloadGeometryData = dataService.downloadIfNotNull(geoDataRef).eagerlyEvaluate(nullptr);
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
            }
                
            Tensor<double, 2> positions = mapTensor<Tensor<double, 2>>(reshapedStartPoints.asReader())
                -> shuffle(Eigen::array<int, 2>{1, 0});
            
            // Create and run the fieldline tracer
            FieldlineTracer tracer(
                device, kernelRequest, field, positions,
                request.getGeometry(), mv(indexData), 
                geometryData.hasValue() ? Maybe<LocalDataRef<MergedGeometry>>(geometryData) : Nothing(),
                mappingData.hasValue() ? Maybe<LocalDataRef<ReversibleFieldlineMapping>>(mappingData) : Nothing(),
                geoMappingData.hasValue() ? Maybe<LocalDataRef<GeometryMapping::MappingData>>(geoMappingData) : Nothing(),
                config
            );
            tracer.run().wait(ws);
            
            // Post-processing
            auto nTurns = computeNTurns(tracer, request, nStartPoints);
            auto nSurfs = request.getPlanes().size();
            auto nRecorded = computeNRecorded(tracer, request, nStartPoints, nSurfs);
            
            auto results = ctx.getResults();
            results.setNTurns(nTurns);
            
            // Initialize and populate result tensors
            populateResults(results, tracer, request, startPointShape, nStartPoints, 
                          nTurns, nSurfs, nRecorded, 
                          geometryData.hasValue() ? geometryData : Nothing(),
                          geoMappingData.hasValue() ? geoMappingData : Nothing());
            
        }).attach(thisCap());
    }
    
    // Helper function to get consolidated data
    kj::Vector<Temporary<FLTKernelData::Entry>> getConsolidatedData(FieldlineTracer& tracer) {
        return tracer.getConsolidatedData();
    }
    
    int64_t computeNTurns(FieldlineTracer& tracer, const FLTRequest& request, int64_t nStartPoints) {
        int64_t nTurns = 0;
        
        auto kData = getConsolidatedData(tracer);
        for(size_t i = 0; i < nStartPoints; ++i) {
            auto entry = kData[i].asReader();
            auto state = entry.getState();
            nTurns = std::max(nTurns, (int64_t) state.getTurnCount() + 1);
        }
        
        if(request.getTurnLimit() > 0)
            nTurns = std::min(nTurns, (int64_t) request.getTurnLimit());
        
        return nTurns;
    }
    
    int64_t computeNRecorded(FieldlineTracer& tracer, const FLTRequest& request, int64_t nStartPoints, int64_t nSurfs) {
        int64_t nPlaneHits = 0;
        auto kData = getConsolidatedData(tracer);
        
        for(int64_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
            auto entry = kData[iStartPoint].asReader();
            auto events = entry.getEvents();
            
            std::vector<int64_t> nHitsPerPlane(nSurfs, 0);
            
            for(auto evt : events) {
                if(evt.isPhiPlaneIntersection()) {
                    auto planeNo = evt.getPhiPlaneIntersection().getPlaneNo();
                    KJ_REQUIRE(planeNo < nSurfs);
                    ++nHitsPerPlane[planeNo];
                }
            }
            
            for(int64_t e : nHitsPerPlane)
                nPlaneHits = std::max(nPlaneHits, e);
        }
        
        int64_t pcLimit;
        switch(request.getPlaneIntersectionRecordMode()) {
            case FLTRequest::PlaneIntersectionRecordMode::EVERY_HIT:
                pcLimit = nPlaneHits;
                break;
            case FLTRequest::PlaneIntersectionRecordMode::LAST_IN_TURN:
                pcLimit = computeNTurns(tracer, request, nStartPoints);
                break;
            default:
                KJ_FAIL_REQUIRE("Unknown plane intersection record mode", request.getPlaneIntersectionRecordMode());
        }
        
        return pcLimit;
    }
    
    void populateResults(
        FLTResponse::Builder results,
        FieldlineTracer& tracer,
        const FLTRequest& request,
        kj::ArrayPtr<const int64_t> startPointShape,
        int64_t nStartPoints,
        int64_t nTurns,
        int64_t nSurfs,
        int64_t nRecorded,
        Maybe<MergedGeometry::Reader> geometryData,
        Maybe<GeometryMapping::MappingData::Reader> geoMappingData
    ) {
        // Get consolidated data from tracer
        auto kData = getConsolidatedData(tracer);
        
        // Initialize tensors
        Tensor<double, 4> pcCuts(nRecorded, nStartPoints, nSurfs, 5);
        Tensor<double, 2> endPoints(nStartPoints, 4);
        Tensor<double, 2> normals(nStartPoints, 3);
        
        fieldline_postprocessing::initializeResultTensors(pcCuts, endPoints, normals);
        
        // Apply shapes
        auto applyPointShape = [&](auto builder, kj::ArrayPtr<const int64_t> preShape, kj::ArrayPtr<const int64_t> postShape) {
            fieldline_postprocessing::applyPointShape(builder, preShape, postShape, startPointShape, nStartPoints);
        };
        
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
        
        size_t nRecordedActual = 0;
        
        for(int64_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
            auto entry = kData[iStartPoint].asReader();
            auto state = entry.getState();
            auto events = entry.getEvents();
            
            // Compute forward and backward free path lengths
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
            std::vector<int64_t> iHitPerPlane(nSurfs, 0);
            
            // Extract Poincaré hits
            fieldline_postprocessing::extractPoincareHits(
                events.asPtr(), pcCuts, nSurfs, nTurns,
                request.getPlaneIntersectionRecordMode(), request
            );
            
            // Extract end points and handle collision
            auto stopReason = entry.getStopReason();
            KJ_REQUIRE(stopReason != FLTStopReason::UNKNOWN, "Internal error: Unknown stop reason encountered", iStartPoint);
            KJ_REQUIRE(stopReason != FLTStopReason::EVENT_BUFFER_FULL, "Internal error: Invalid stop reason EVENT_BUFFER_FULL", iStartPoint);
            
            stopReasonData.set(iStartPoint, stopReason);
            stepCountData.set(iStartPoint, state.getNumSteps());
            
            // Get end point position
            auto stopLoc = state.getPosition();
            for(int i = 0; i < 3; ++i) {
                endPoints(iStartPoint, i) = stopLoc[i];
            }
            endPoints(iStartPoint, 3) = state.getDistance();
            
            // Handle collision limits
            if(stopReason == FLTStopReason::COLLISION_LIMIT) {
                KJ_IF_MAYBE(pLastCollision, lastCollision) {
                    auto& lastCollision = *pLastCollision;
                    
                    // Determine appropriate geometry to use for the mesh analysis
                    auto getGeometrySource = [&]() -> MergedGeometry::Reader {
                        auto geoSource = lastCollision.getGeometryHit().getGeometrySource();
                        
                        if(geoSource.isBaseGeometry()) {
                            KJ_IF_MAYBE(pGeometryData, geometryData) {
                                return pGeometryData -> get();
                            }
                            KJ_FAIL_REQUIRE("Internal error: Base geometry not present despite collision with it");
                        } else if(geoSource.isMappingSection()) {
                            KJ_IF_MAYBE(pGeoMappingData, geoMappingData) {
                                auto sections = pGeoMappingData -> get().getSections();
                                return sections[geoSource.getMappingSection() % sections.size()].getGeometry();
                            }
                            KJ_FAIL_REQUIRE("Internal error: Mapping geometry not present despite collision with it");
                        } else {
                            KJ_FAIL_REQUIRE("Unknown collision geometry source");
                        }
                    };
                    
                    MergedGeometry::Reader geometry = getGeometrySource();
                    
                    uint32_t meshIdx = lastCollision.getGeometryHit().getMeshIndex();
                    uint32_t elementIdx = lastCollision.getGeometryHit().getElementIndex();
                    
                    auto tagValues = geometry.getEntries()[meshIdx].getTags();
                    
                    for(auto iTag : kj::indices(tagValues)) {
                        endTagData.setWithCaveats(iTag * nStartPoints + iStartPoint, tagValues[iTag]);
                    }
                    
                    auto mesh = geometry.getEntries()[meshIdx].getMesh();
                    
                    uint32_t offset = 0;
                    bool skip = false;
                    switch(mesh.which()) {
                        case Mesh::TRI_MESH:
                            offset = 3 * elementIdx;
                            break;
                        case Mesh::POLY_MESH: {
                            offset = mesh.getPolyMesh()[elementIdx];
                            auto next = mesh.getPolyMesh()[elementIdx + 1];
                            
                            // Skip elements with less than 3 points
                            if(next < offset + 3)
                                skip = true;
                            
                            break;
                        }
                        
                        default:
                            KJ_FAIL_REQUIRE("Unknown mesh type", mesh.which());
                    }
                    
                    if(!skip) {
                        auto data = mesh.getVertices().getData();
                        Eigen::Vector3d vertices[3];
                        for(int i = 0; i < 3; ++i) {
                            uint32_t index = mesh.getIndices()[offset + i];
                            
                            vertices[i] = Eigen::Vector3d(
                                data[3 * index], data[3 * index + 1], data[3 * index + 2]
                            );
                        }
                        
                        // If the geometry source is a section, we need to transform the triangle
                        // back to regular mapping space.
                        
                        if(lastCollision.getGeometryHit().getGeometrySource().isMappingSection()) {
                            // Note: Need access to mapping data for this transformation
                            // For now, skip the transformation
                        }
                        
                        Eigen::Vector3d normal = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[1]);
                        normal /= normal.norm();
                        
                        for(int i = 0; i < 3; ++i) {
                            normals(iStartPoint, i) = normal(i);
                        }
                    }
                }
            }
            
            // Track recorded points
            for(auto evt : events) {
                if(evt.isRecord()) {
                    ++recordedForThis;
                }
            }
            
            nRecordedActual = kj::max(nRecordedActual, recordedForThis);
        }
        
        // Write tensors
        writeTensor(pcCuts, results.getPoincareHits());
        applyPointShape(results.getPoincareHits(), {5, nSurfs}, {nRecorded});
        
        writeTensor(endPoints, results.getEndPoints());
        applyPointShape(results.getEndPoints(), {4}, {});
        
        writeTensor(normals, results.getNormals());
        applyPointShape(results.getNormals(), {3}, {});
        
        // Handle recorded field lines
        if(nRecordedActual > 0) {
            KJ_REQUIRE(
                3 * nRecordedActual * nStartPoints * capnp::ELEMENTS <= capnp::MAX_LIST_ELEMENTS,
                "The number of recorded points is so large that the returned"
                " tensor would exceed size limitations. Please reduce the number"
                " of points, increase step size, or increase no. of steps between"
                " recordings"
            );
            
            Tensor<double, 3> fieldLines(nRecordedActual, nStartPoints, 3);
            Tensor<double, 2> fieldStrengths(nRecordedActual, nStartPoints);
            
            fieldLines.setConstant(std::nan(""));
            fieldStrengths.setConstant(std::nan(""));
            
            auto kData = getConsolidatedData(tracer);
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
            
            applyPointShape(results.getFieldLines(), {3}, {(int64_t) nRecordedActual});
            applyPointShape(results.getFieldStrengths(), {}, {(int64_t) nRecordedActual});
        }
        
        // Calculate iota if requested
        if(request.getFieldLineAnalysis().isCalculateIota()) {
            auto iotas = results.getFieldLineAnalysis().initIotas();
            applyPointShape(iotas, {}, {});
            auto iotaData = iotas.getData();
            
            auto kData = getConsolidatedData(tracer);
            for(auto iStartPoint : kj::indices(iotaData)) {
                auto state = kData[iStartPoint].getState();
                
                double myIota = state.getTheta() / state.getPhi();
                iotaData.set(iStartPoint, myIota);
            }
        }
        
        // Calculate Fourier modes if requested
        if(request.getFieldLineAnalysis().isCalculateFourierModes()) {
            auto calcFM = request.getFieldLineAnalysis().getCalculateFourierModes();
            KJ_REQUIRE(calcFM.getIota().getData().size() == nStartPoints, "Incorrect iota count specified");
            
            for(int64_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
                auto kData = getConsolidatedData(tracer);
                auto entry = kData[iStartPoint].asReader();
                auto events = entry.getEvents();
                
                const double iota = calcFM.getIota().getData()[iStartPoint];
                
                fieldline_postprocessing::calculateFourierModes(
                    events.asPtr(), iota, calcFM, results
                );
            }
        }
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
        Shared<Tensor<double, 2>> points;
        auto vardimShape = readVardimTensor(ctx.getParams().getPoints(), 1, *points);
        size_t nPoints = points -> dimension(0);
        
        struct IterResult {
            double r;
            double z;
        };
        
        auto params = ctx.getParams().getRequest();
        auto nPhi = params.getNPhi();
        auto islandM = params.getIslandM();
        
        Shared<Tensor<double, 3>> axis(nPhi * islandM, nPoints, 3);
        Shared<Tensor<double, 1>> fields(nPoints);
        Shared<Tensor<double, 2>> pos(nPoints, 3);
        
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
        .attach(mv(field));
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

} // namespace fsc

// Factory function
namespace fsc {
Own<FLT::Server> newFLT(Own<DeviceBase> device, FLTConfig::Reader config) {
    return kj::heap<FLTImpl>(mv(device), config);
}
}
