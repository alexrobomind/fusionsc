#include "fieldline-postprocessing.h"

#include <algorithm>
#include <limits>

namespace fsc {
namespace fieldline_postprocessing {

std::array<double, 3> getEventLocation(FLTKernelEvent::Reader e) {
    return {e.getX(), e.getY(), e.getZ()};
}

template<typename BuilderType>
void applyPointShape(
    BuilderType& builder,
    kj::ArrayPtr<const int64_t> preShape,
    kj::ArrayPtr<const int64_t> postShape,
    kj::ArrayPtr<const int64_t> startPointShape,
    int64_t nStartPoints
) {
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
}

void initializeResultTensors(
    Tensor<double, 4>& pcCuts,
    Tensor<double, 2>& endPoints,
    Tensor<double, 2>& normals
) {
    pcCuts.setConstant(std::numeric_limits<double>::quiet_NaN());
    endPoints.setConstant(std::numeric_limits<double>::quiet_NaN());
    normals.setConstant(std::numeric_limits<double>::quiet_NaN());
}

void extractPoincareHits(
    kj::ArrayPtr<FLTKernelEvent::Reader> events,
    Tensor<double, 4>& pcCuts,
    int64_t nSurfs,
    int64_t nTurns,
    FLTRequest::PlaneIntersectionRecordMode recordMode,
    const FLTRequest& request
) {
    std::vector<int64_t> iHitPerPlane(nSurfs, 0);
    
    int64_t iTurn = 0;
    double lastNewTurnDistance = 0;
    
    for(auto iEvt : kj::indices(events)) {
        auto evt = events[iEvt];
        
        KJ_REQUIRE(!evt.isNotSet(), "Internal error: Event not set");
                        
        if(evt.isNewTurn()) {
            iTurn = evt.getNewTurn();
            lastNewTurnDistance = evt.getDistance();
        } else if(evt.isPhiPlaneIntersection()) {
            auto ppi = evt.getPhiPlaneIntersection();
            
            int64_t iTurnActual = iTurn;
            
            if(recordMode == FLTRequest::PlaneIntersectionRecordMode::LAST_IN_TURN) {
                // Adjustments to get consistent behavior for turn recording
                
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
            } else {
                // We record every plane hit.
                auto planeNo = ppi.getPlaneNo();
                KJ_REQUIRE(planeNo < nSurfs);
                iTurnActual = iHitPerPlane[planeNo]++;
            }
            
            auto loc = getEventLocation(evt);
            for(int64_t iDim = 0; iDim < 3; ++iDim) {
                pcCuts(iTurnActual, 0, ppi.getPlaneNo(), iDim) = loc[iDim];
            }
            pcCuts(iTurnActual, 0, ppi.getPlaneNo(), 3) = /* forwardLCs[iEvt] */ 0.0;
            pcCuts(iTurnActual, 0, ppi.getPlaneNo(), 4) = /* backwardLCs[iEvt] */ 0.0;
        }
    }
}

void calculateIota(
    FLTKernelData::Entry::State::Reader state,
    Float64Tensor::Builder iotas
) {
    double myIota = state.getTheta() / state.getPhi();
    iotas.setData({myIota});
}

void calculateFourierModes(
    kj::ArrayPtr<FLTKernelEvent::Reader> events,
    double iota,
    FLTRequest::FieldLineAnalysis::CalculateFourierModes::Reader calcFM,
    capnp::StructBuilder& resultBuilder
) {
    using FP = nudft::FourierPoint<2, 2>;
    using FM = nudft::FourierMode<2, 2>;
    
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
    
    Tensor<double, 3> rCos(nPoloicalCoeffs, nToroidalCoeffs, 1);
    Tensor<double, 3> rSin(nPoloicalCoeffs, nToroidalCoeffs, 1);
    Tensor<double, 3> zCos(nPoloicalCoeffs, nToroidalCoeffs, 1);
    Tensor<double, 3> zSin(nPoloicalCoeffs, nToroidalCoeffs, 1);
    
    Tensor<double, 2> nTor(nPoloicalCoeffs, nToroidalCoeffs);
    Tensor<double, 2> mPol(nPoloicalCoeffs, nToroidalCoeffs);
    
    rCos.setConstant(0);
    rSin.setConstant(0);
    zCos.setConstant(0);
    zSin.setConstant(0);
    nTor.setConstant(0);
    mPol.setConstant(0);
    
    // Index of n = 0, m = 1 mode
    const size_t n0m1Index = 1;
    
    kj::Vector<FM> modes;
    
    for(auto in : kj::range(0, nToroidalCoeffs)) {
        auto n = in <= maxN ? in : in - nToroidalCoeffs;
        
        for(auto m : kj::range(0, maxM + 1)) {
            nTor(m, toroidalIndex(n)) = n * phiMultiplier;
            mPol(m, toroidalIndex(n)) = m;
            
            double parallelAngle = std::abs(n * phiMultiplier + m * iota);
            
            if(m == 0 && n < 0) {
                continue;
            }
            
            kj::Maybe<FM&> modeAliases = nullptr;
            for(auto& prevMode : modes) {
                double prevParAngle = std::abs(prevMode.coeffs[0] * phiMultiplier + prevMode.coeffs[1] * iota);
                
                if(std::abs(parallelAngle - prevParAngle) < aliasThreshold) {
                    modeAliases = prevMode;
                }
            }
            
            KJ_IF_MAYBE(pOther, modeAliases) {
                const int on = pOther -> coeffs[0];
                const int om = pOther -> coeffs[1];
                
                bool keepMe = m < om;
                bool keepOther = !keepMe;
                
                if((m == 0 && n == 0) || (m == 1 && n == 0)) keepMe = true;
                if((om == 0 && on == 0) || (om == 1 && on == 0)) keepOther = true;
                
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
    
    // Subtract theta0 from poloidal angle
    for(auto& p : fourierPoints) {
        p.angles[1] -= theta0;
    }
    
    // Now do the full Fourier calculation
    nudft::calculateModes<2, 2>(fourierPoints.asPtr(), modes);
    
    for(auto mode : modes) {
        rCos(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), 0) = mode.cosCoeffs[0];
        zSin(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), 0) = mode.sinCoeffs[1];
        rSin(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), 0) = mode.sinCoeffs[0];
        zCos(mode.coeffs[1], toroidalIndex(mode.coeffs[0]), 0) = mode.cosCoeffs[1];
    }
    
    auto out = resultBuilder.initFieldLineAnalysis().initFourierModes();
    
    auto surf = out.initSurfaces();
    surf.setNTor(maxN);
    surf.setMPol(maxM);
    surf.setToroidalSymmetry(calcFM.getToroidalSymmetry());
    surf.setNTurns(calcFM.getIslandM());
    // writeTensor(rCos, surf.getRCos());
    // writeTensor(zSin, surf.getZSin());
    
    applyPointShape(surf.getRCos(), {}, {nToroidalCoeffs, nPoloicalCoeffs}, {}, 1);
    applyPointShape(surf.getZSin(), {}, {nToroidalCoeffs, nPoloicalCoeffs}, {}, 1);
    
    if(!calcFM.getStellaratorSymmetric()) {
        auto ns = surf.initNonSymmetric();
        // writeTensor(rSin, ns.getRSin());
        // writeTensor(zCos, ns.getZCos());
        applyPointShape(ns.getRSin(), {}, {nToroidalCoeffs, nPoloicalCoeffs}, {}, 1);
        applyPointShape(ns.getZCos(), {}, {nToroidalCoeffs, nPoloicalCoeffs}, {}, 1);
    }
    
    auto oNTor = out.initNTor(2 * maxN + 1);
    for(int i : kj::indices(oNTor)) {
        oNTor.set(i, -nTor(0, i));
    }
    
    auto oMPol = out.initMPol(maxM + 1);
    for(int i : kj::indices(oMPol))
        oMPol.set(i, mPol(i, 0));
}

} // namespace fieldline_postprocessing
} // namespace fsc
