#pragma once

#include <fsc/flt.capnp.cu.h>

#include "kernels/kernels.h"

#include "tensor.h"
#include "kernels/tensor.h"
#include "kernels/message.h"

#include "interpolation.h"
#include "intersection.h"
#include "geometry.h"
#include "random-kernels.h"
#include "fieldline-mapping.h"


namespace fsc {

FSC_DECLARE_KERNEL(
	fltKernel,
	
	cu::FLTKernelData::Builder,
	Eigen::TensorMap<Eigen::Tensor<double, 4>>,
	cu::FLTKernelRequest::Builder,
	
	cu::MergedGeometry::Reader,
	cu::IndexedGeometry::Reader,
	cu::IndexedGeometry::IndexData::Reader,
	
	cu::ReversibleFieldlineMapping::Reader
);
	
namespace kmath {
	template<typename Num>
	EIGEN_DEVICE_FUNC Num wrap(Num x) { 
		x = fmod(x, 2 * pi);
		x += 3 * pi;
		x = fmod(x, 2 * pi);
		x -= pi;
		return x;
	};
	
	// Check if the (phi1, phi2] interval crosses across phi0.
	template<typename Num>	
	EIGEN_DEVICE_FUNC bool crossedPhi(Num phi1, Num phi2, Num phi0) {
		auto d1 = wrap(phi0 - phi1);
		auto d2 = wrap(phi2 - phi1);
		
		if(d1 > 0 && d2 > 0 && d1 <= d2)
			return true;
		
		if(d1 < 0 && d2 < 0 && d1 >= d2)
			return true;
		
		return false;
	};
	
	template<typename Num, int dim, typename F>
	EIGEN_DEVICE_FUNC void runge_kutta_4_step(Eigen::Vector<Num, dim>& x, Num t, Num h, const F& f) {
		using XType = Eigen::Vector<Num, dim>;

		std::array<XType, 4> directions;

		directions[0] = f(x                          , t          );
		directions[1] = f(x + 0.5 * h * directions[0], t + 0.5 * h);
		directions[2] = f(x + 0.5 * h * directions[1], t + 0.5 * h);
		directions[3] = f(x +       h * directions[2], t +       h);

		x += h * (
			1.0 / 6.0 * (directions[0] + directions[3]) +
			1.0 / 3.0 * (directions[1] + directions[2])
		);
	}
	
	template<typename Num, int dim, typename F>
	EIGEN_DEVICE_FUNC double runge_kutta_fehlberg_step(Eigen::Vector<Num, dim>& x, Num t, Num h, const F& f) {
		using XType = Eigen::Vector<Num, dim>;

		std::array<XType, 6> directions;
		
		directions[0] = f(x, t);
		directions[1] = f(
			x + 0.25 * h * directions[0],
			t + 0.25 * h
		);
		
		directions[2] = f(
			x + 3./32 * h * directions[0]
			  + 9./32 * h * directions[1],
			t + 3./8 * h
		);
		directions[3] = f(
			x + 1932./2197 * h * directions[0]
			  - 7200./2197 * h * directions[1]
			  + 7296./2197 * h * directions[2],
			t + 12./13 * h
		);
		directions[4] = f(
			x + 439./216 * h * directions[0]
			  - 8        * h * directions[1]
			  + 3680./513 * h * directions[2]
			  - 845./4104 * h * directions[3],
			t + h
		);
		directions[5] = f(
			x - 8./27 * h * directions[0]
			  + 2 * h * directions[1]
			  - 3544./2565 * h * directions[2]
			  + 1859./4104 * h * directions[3]
			  - 11./40 * h * directions[4],
			 t + 0.5 * h
		);
		
		XType fifthOrder =
			  16./135 * h * directions[0]
			+ 6656./12825 * h * directions[2]
			+ 28561./56430 * h * directions[3]
			- 9./50 * h * directions[4]
			+ 2./55 * h * directions[5];
		
		XType fourthOrder =
			  25./216 * h * directions[0]
			+ 1408./2565 * h * directions[2]
			+ 2197./4104 * h * directions[3]
			- 1./5 * h * directions[4];
		
		x += fifthOrder;
		return (fifthOrder - fourthOrder).norm();
	}
}


/*

Some special programming rules when working inside this function:

- The function uses a goto-block for termination which is equivalent to a try-finally construct.
  Therefore, NEVER use "return" to leave. Always use the FSC_FLT_RETURN macro, which will also
  carry the termination reason.
- The fieldline tracer uses a log buffer to record special events such as Poincare intersections,
  direction reversals, or collisions.

*/

/**
 \ingroup kernels
 */
EIGEN_DEVICE_FUNC inline void fltKernel(
	unsigned int idx,
	
	cu::FLTKernelData::Builder kernelData,
	TensorMap<Tensor<double, 4>> fieldData,
	cu::FLTKernelRequest::Builder kernelRequest,
	
	cu::MergedGeometry::Reader geometry,
	cu::IndexedGeometry::Reader index,
	cu::IndexedGeometry::IndexData::Reader indexData,
	
	cu::ReversibleFieldlineMapping::Reader flmData
) {
	using Num = double;
	using V3 = Vec3<Num>;
	using V2 = Vec2<Num>;
	using V1 = Vec1<Num>;
	
	auto request = kernelRequest.getServiceRequest();
	
	auto parModel = request.getParallelModel();
	auto perpModel = request.getPerpendicularModel();
	
	bool useFLM = flmData.getSections().size() > 0;
	
	// printf("Hello there\n");
	//CUPNP_DBG("FLT kernel started", idx, useFLM);
	
	// Extract local scratch space
	auto myData = kernelData.mutateData()[idx];
	auto state = myData.mutateState();
	auto statePos = state.mutatePosition();
	
	// Copy initial state data into local memory
	V3 x;
	for(int i = 0; i < 3; ++i)
		x[i] = statePos[i];
	
	uint32_t step = state.getNumSteps();
	uint32_t eventCount = state.getEventCount();
	uint32_t turn = state.getTurnCount();
	Num distance = state.getDistance();
	double nextDisplacementStep = state.getNextDisplacementAt();
	uint32_t displacementCount = state.getDisplacementCount();
	MT19937 rng(state.getRngState());
	
	// Set up the magnetic field
	//using InterpolationStrategy = LinearInterpolation<Num>;
	using InterpolationStrategy = C1CubicInterpolation<Num>;
	
	auto grid = request.getField().getGrid();
	SlabFieldInterpolator<InterpolationStrategy> interpolator(InterpolationStrategy(), grid);
	
	// Set up the magnetic axis
	cupnp::List<double>::Reader rAxis(0, nullptr);
	cupnp::List<double>::Reader zAxis(0, nullptr);
	auto rAxisAt = [&](int i) {
		i %= rAxis.size();
		i += rAxis.size();
		i %= rAxis.size();
		return rAxis[i];
	};
	auto zAxisAt = [&](int i) {
		i %= zAxis.size();
		i += zAxis.size();
		i %= zAxis.size();
		return zAxis[i];
	};
	
	// Set up unwrapping configuration
	const double iota = state.getIota();
	double theta = state.getTheta();
	uint32_t unwrapEvery = 0;
	
	using AxisInterpolator = NDInterpolator<1, C1CubicInterpolation<double>>;
	AxisInterpolator axisInterpolator(C1CubicInterpolation<double>(), { AxisInterpolator::Axis(0, 2 * pi, rAxis.size()) });
	
	{
		auto fla = request.getFieldLineAnalysis();
		if(fla.isCalculateIota()) {
			rAxis = fla.getCalculateIota().getRAxis();
			zAxis = fla.getCalculateIota().getZAxis();
			unwrapEvery = fla.getCalculateIota().getUnwrapEvery();
		} else if(fla.isCalculateFourierModes()) {
			rAxis = fla.getCalculateFourierModes().getRAxis();
			zAxis = fla.getCalculateFourierModes().getZAxis();
		}
	}
	
	bool processDisplacements = perpModel.hasIsotropicDiffusionCoefficient() || perpModel.hasRzDiffusionCoefficient();
	
	// Compute tracing and field orientation
	// Do we trace forward (+1) or backward (-1) ?
	int8_t tracingDirection = state.getForward() ? 1 : -1;
	
	// Does the field point in CCW (+1) or CW (-1) direction ?
	int8_t fieldOrientation;
	{
		double phi = atan2(x[1], x[0]);
		V3 fieldValue = interpolator(fieldData, x);
		fieldOrientation = (fieldValue[1] * cos(phi) - fieldValue[0] * sin(phi)) > 0 ? 1 : -1;
	}
	
	int8_t forwardDirection;
	if(request.getForwardDirection().hasField()) {
		forwardDirection = fieldOrientation;
	} else {
		forwardDirection = 1;
	}
	
	int8_t forwardDirectionRelField = forwardDirection * fieldOrientation;
	
	auto rungeKuttaInput = [&](V3 x, Num t) -> V3 {
		V3 fieldValue = interpolator(fieldData, x);
		auto result = fieldValue / fieldValue.norm() * tracingDirection * forwardDirectionRelField;
		
		/*if(step % 10 == 0) {
			KJ_DBG(result[0], result[1], result[2], result.norm());
		}*/
		return result;
	};
	
	// The kernel terminates its execution with this macro
	#define FSC_FLT_RETURN(reason) {\
		myData.setStopReason(::fsc::cu::FLTStopReason::reason); \
		goto THE_END; \
	}
	
	// Event logging
	// As this involves our return macro, it can't go
	// into a lambda
	#define FSC_FLT_LOG_EVENT(x, evtDist) {\
		if(eventCount >= myData.getEvents().size() - 1) { \
			FSC_FLT_RETURN(EVENT_BUFFER_FULL); \
		} \
		\
		{\
			auto evt = currentEvent(); \
			evt.setStep(step); \
			evt.setDistance(evtDist); \
			evt.setX(x[0]); \
			evt.setY(x[1]); \
			evt.setZ(x[2]); \
		}\
		\
		++eventCount; \
	}
	
	auto currentEvent = [&]() {
		return myData.mutateEvents()[eventCount];
	};
	
	RFLM flm(flmData);
	
	// Initialize mapped position
	if(useFLM) {
		flm.map(x, tracingDirection * forwardDirection > 0);
		
		/*if(idx == 0) {
			V3 x2 = flm.unmap(flm.phi);
			KJ_DBG(step);
			KJ_DBG(x[0], x2[0]);
			KJ_DBG(x[1], x2[1]);
			KJ_DBG(x[2], x2[2]);
		}*/
	}
	
	// ... do the work ...
	
	while(true) {		
		// KJ_DBG("Beginning loop", step, distance);
		
		// Check various limits
		if(step >= request.getStepLimit() && request.getStepLimit() > 0)
			FSC_FLT_RETURN(STEP_LIMIT);
		
		if(distance >= request.getDistanceLimit() && request.getDistanceLimit() > 0)
			FSC_FLT_RETURN(DISTANCE_LIMIT);
		
		if(turn >= request.getTurnLimit() && request.getTurnLimit() > 0)
			FSC_FLT_RETURN(TURN_LIMIT);
		
		if(x != x)
			FSC_FLT_RETURN(NAN_ENCOUNTERED);
		
		// Position recording
		if(request.getRecordEvery() != 0 && (step % request.getRecordEvery() == 0)) {
			auto rec = currentEvent().mutateRecord();
			V3 fv = interpolator(fieldData, x);
			rec.setFieldStrength(std::sqrt(fv[0] * fv[0] + fv[1] * fv[1] + fv[2] * fv[2]));
			
			FSC_FLT_LOG_EVENT(x, distance)
		}
					
		Num r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
		Num z = x[2];
		
		// KJ_DBG(r, z);
		
		// Unwrapping of phase
		if(unwrapEvery != 0 && (step % unwrapEvery == 0)) {
			double phi = atan2(x[1], x[0]);
			double rAxisVal = axisInterpolator(rAxisAt, V1(phi));
			double zAxisVal = axisInterpolator(zAxisAt, V1(phi));
			
			double dr = r - rAxisVal;
			double dz = z - zAxisVal;
			
			double newTheta = atan2(dz, dr);
			double dTheta = newTheta - theta;
			dTheta = fmod(dTheta + pi, 2 * pi) - pi;
			
			theta += dTheta;
		}
		
		// KJ_DBG("In grid?", r, z, grid.getRMin(), grid.getRMax(), grid.getZMin(), grid.getZMax());
		
		if(r <= grid.getRMin() || r >= grid.getRMax() || z <= grid.getZMin() || z >= grid.getZMax()) {
			FSC_FLT_RETURN(OUT_OF_GRID);
		}
		
		// KJ_DBG("Limits passed");
		
		V3 x2 = x;
		
		bool displacementStep = false;
		
		if(processDisplacements) {
			if(distance > nextDisplacementStep)
				displacementStep = true;
		}
		
		// KJ_DBG(displacementStep);
		double stepSize = state.getStepSize();
		
		if(displacementStep) {
			double prevFreePath = parModel.getMeanFreePath() + displacementCount * parModel.getMeanFreePathGrowth();
			double nextFreePath = parModel.getMeanFreePath() + (1 + displacementCount) * parModel.getMeanFreePathGrowth();
			
			// Sample some normal distributed numbers
			double normalDistributed[4];
			
			// KJ_DBG("Running rng", idx);
			rng.normalPair(normalDistributed[0], normalDistributed[1]);
			rng.normalPair(normalDistributed[2], normalDistributed[3]);
			
			// To be filled out by transport model
			double deltaT = 0;
			double freePath = 0;
			
			if(parModel.hasConvectiveVelocity()) {
				// Convective transport model
				deltaT = prevFreePath / parModel.getConvectiveVelocity();					
				freePath = rng.exponential() * nextFreePath;
				
			} else if(parModel.hasDiffusionCoefficient()) {
				// Diffusive transport model
				// TODO: Check prefactor
				deltaT = prevFreePath * prevFreePath / parModel.getDiffusionCoefficient();
				freePath = normalDistributed[3] * nextFreePath;
			}
			// KJ_DBG(idx, deltaT, freePath);
			
			if(freePath >= 0) {
				nextDisplacementStep += freePath;
			} else {
				tracingDirection = -tracingDirection;
				nextDisplacementStep -= freePath;
			}
				
			// Perform displacement
			
			if(perpModel.hasIsotropicDiffusionCoefficient()) {
				double isoDisplacement = std::sqrt(deltaT * perpModel.getIsotropicDiffusionCoefficient());
				// KJ_DBG(idx, isoDisplacement);
								
				for(int i = 0; i < 3; ++i) {
					x2[i] += isoDisplacement * normalDistributed[i];
				}
			} else if(perpModel.hasRzDiffusionCoefficient()) {
				double rzDisplacement = std::sqrt(deltaT * perpModel.getRzDiffusionCoefficient());
				
				double dr = rzDisplacement * normalDistributed[0];
				double dz = rzDisplacement * normalDistributed[1];
				
				double invR = 1 / r;
				x2[0] += x2[0] * invR * dr;
				x2[1] += x2[1] * invR * dr;
				x2[2] += dz;
			}
			
			++displacementCount;
			
			if(useFLM) {
				flm.map(x2, tracingDirection * forwardDirection > 0);
			}				
		} else {			
			auto controlInfo = request.getStepSizeControl();
			if(controlInfo.hasFixed()) {
				if(processDisplacements) {
					stepSize = std::min(stepSize, nextDisplacementStep - distance + 0.001);
				}
				
				if(useFLM) {
					flm.setFieldlinePosition(0);
					double newPhi = flm.phi + forwardDirection * tracingDirection * stepSize / r;
					x2 = flm.advance(newPhi);
				} else {
					// Regular tracing step
					kmath::runge_kutta_4_step(x2, .0, stepSize, rungeKuttaInput);
				}
			} else {
				auto adaptiveInfo = controlInfo.getAdaptive();
				double maxVal = adaptiveInfo.getMax();
				double minVal = adaptiveInfo.getMin();
				double targetError = adaptiveInfo.getTargetError();
				
				if(processDisplacements) {
					maxVal = std::min(maxVal, nextDisplacementStep - distance + 0.001);
				}
				
				while(true) {
					stepSize = std::min(std::max(stepSize, minVal), maxVal);
					
					// Perform Runge-Kutta-Fehlmann step
					double errorEstimate = kmath::runge_kutta_fehlberg_step(x2, .0, stepSize, rungeKuttaInput);
					
					// Try to adapt step size (assuming 4th order convergence)
					double prevStepSize = stepSize;
					
					if(adaptiveInfo.getErrorUnit().hasStep()) {
						stepSize *= std::pow(adaptiveInfo.getTargetError() / errorEstimate, 0.2);
					} else {
						errorEstimate *= adaptiveInfo.getErrorUnit().getIntegratedOver() / stepSize;
						stepSize *= std::pow(adaptiveInfo.getTargetError() / errorEstimate, 0.25);
					}
					
					// KJ_DBG(prevStepSize, errorEstimate, stepSize);
					
					// Check whether we need to re-run the step
					if(errorEstimate > adaptiveInfo.getTargetError() * (1 + adaptiveInfo.getRelativeTolerance()) && prevStepSize > minVal) {
						// Reset step
						x2 = x;
						continue;
					} else {
						break;
					}
				}
			}
		}
		
		// KJ_DBG("Step advanced", x[0], x[1], x[2], x2[0], x2[1], x2[2]);
		// KJ_DBG("|dx|", (x2 - x).norm());
					
		// --- Check for plane crossings ---
		
		Num phi1 = atan2(x[1], x[0]);
		Num phi2 = atan2(x2[1], x2[0]);
		
		/*if(step % 1000000 == 0) {
			KJ_DBG(step, distance, phi1, z, r);
		}*/
		
		Num phi0 = state.getPhi0();
		
		const auto planes = request.getPlanes();
		for(uint32_t iPlane = 0; iPlane < planes.size(); ++iPlane) {
			const auto plane = planes[iPlane];
			const auto orientation = plane.getOrientation();
			
			bool crossed = false;
			double crossedAt = 0;
			
			if(orientation.hasPhi()) {
				double planePhi = orientation.getPhi();
			
				if(kmath::crossedPhi(phi1, phi2, planePhi)) {
					crossedAt = kmath::wrap(planePhi - phi1) / kmath::wrap(phi2 - phi1);
					crossed = true;
				}
			} else if(orientation.hasNormal()) {
				Vec3d center(0, 0, 0);
				
				auto pCenter = plane.getCenter();
				if(pCenter.size() == 3) {
					center[0] = pCenter[0];
					center[1] = pCenter[1];
					center[2] = pCenter[2];
				}
				
				Vec3d normal(0, 0, 0);
				auto pNormal = orientation.getNormal();
				if(pNormal.size() == 3) {
					normal[0] = pNormal[0];
					normal[1] = pNormal[1];
					normal[2] = pNormal[2];
				}
				
				double d1 = (x - center).dot(normal);
				double d2 = (x2 - center).dot(normal);
				
				if(d1 < 0 && d2 >= 0) {
					crossedAt = (0 - d1) / (d2 - d1);
					crossed = true;
				}
			}
			
			if(crossed) {
				V3 xCross = crossedAt * x2 + (1. - crossedAt) * x;
				
				// If we use field-line mapping, the linear interpolation might be unreasonable
				// due to large step sizes. Instead, use the mapping's spline interpolation
				double crossDist = 0;
				if(useFLM && !displacementStep && orientation.hasPhi()) {
					RFLM flm2 = flm;
					double phiCross = (1 - crossedAt) * kmath::wrap(phi1 - phi2) + flm2.phi;
					
					xCross = flm2.advance(phiCross);
					crossDist = fabs(flm2.getFieldlinePosition(phiCross));
				} else {
					crossDist = (xCross - x).norm();
				}
				
				currentEvent().mutatePhiPlaneIntersection().setPlaneNo(iPlane);
									
				FSC_FLT_LOG_EVENT(xCross, distance + crossDist);				
			}
		}
		
		// --- Check for new turns ---
					
		if(kmath::crossedPhi(phi1, phi2, phi0) && step > 1) {
			auto l = kmath::wrap(phi0 - phi1) / kmath::wrap(phi2 - phi1);
			V3 xCross = l * x2 + (1. - l) * x;
			
			// Same as above with the usual planes, interpolate if we are using the mapping
			double crossDist = 0;
			if(useFLM && !displacementStep) {
				RFLM flm2 = flm;
				// KJ_DBG("Pre advance", flm2.uv(0), flm2.uv(1));
				xCross = flm2.advance(flm2.phi + kmath::wrap(phi0 - phi2));
				// KJ_DBG("Post advance", flm2.uv(0), flm2.uv(1));
				crossDist = flm2.getFieldlinePosition(phi1 + kmath::wrap(phi0 - phi1));
			} else {
				crossDist = (xCross - x).norm();
			}
			
			// printf("New turn\n");
			
			// KJ_DBG(idx, state.getTurnCount());
			
			currentEvent().setNewTurn(turn + 1);
			FSC_FLT_LOG_EVENT(xCross, distance + crossDist);		
			
			++turn;		
		}
		
		// --- Check for collisions ---
		
		uint32_t numCollisions = 0;
		
		if(indexData.getGridContents().getData().size() > 0) {
			auto eventBuffer = myData.mutateEvents();
			uint32_t newEventCount = intersectGeometryAllEvents(x, x2, geometry, index, indexData, 1, eventBuffer, eventCount);
							
			if(newEventCount == eventBuffer.size()) {
				FSC_FLT_RETURN(EVENT_BUFFER_FULL);
			}
			
			numCollisions = newEventCount - eventCount;
			
			for(auto iEvt = eventCount; iEvt < newEventCount; ++iEvt) {
				auto curEvt = eventBuffer[iEvt];
				curEvt.setDistance(distance + curEvt.getDistance());
				curEvt.setStep(step);
			}
			
			eventCount = newEventCount;
		}
					
		// KJ_DBG("Phi cross checks passed");
		
		// --- Sort generated events ---
		{
			auto events = myData.mutateEvents();
			
			// TODO: This is slow as hell
			for(auto i1 = state.getEventCount(); i1 < eventCount; ++i1) {
				for(auto i2 = i1 + 1; i2 < eventCount; ++i2) {
					auto event1 = events[i1];
					auto event2 = events[i2];
					
					if(event1.getDistance() > event2.getDistance()) {
						cupnp::swapData(event1, event2);
					}
				}
			}
		}
		
		// Check if the field line needs to be interrupted in-between events
		// Currently, this is only the case for mid-flight final collisions.
		
		const uint32_t collisionLimit = request.getCollisionLimit();
		if(collisionLimit != 0 && state.getCollisionCount() + numCollisions >= collisionLimit) {
			uint32_t collisionCounter = state.getCollisionCount();
			uint32_t eventOffset = state.getEventCount();
			
			auto events = myData.mutateEvents();
			
			while(eventOffset < eventCount) {					
				auto evt = events[eventOffset];
				
				// Note: This would be the point to also check for other termination criteria
				if(evt.isGeometryHit()) {
					if(++collisionCounter >= collisionLimit) {
						// We have found our final event
						// Copy out distance and location and finish
						x[0] = evt.getX();
						x[1] = evt.getY();
						x[2] = evt.getZ();
						distance = evt.getDistance();
						
						++eventOffset;
						break;
					}
				}
				
				++eventOffset;
			}
			
			eventCount = eventOffset;
			state.setEventCount(eventCount);
			state.setCollisionCount(collisionLimit);
			
			FSC_FLT_RETURN(COLLISION_LIMIT);
		}
		
		// --- Advance the step after all events are processed ---
		
		if(!displacementStep) {
			if(useFLM) {
				distance += fabs(flm.getFieldlinePosition(flm.phi));
			} else {
				distance += stepSize;
			}
		} else {
			distance += (x2 - x).norm();
		}
		
		x = x2;
		
		state.setEventCount(eventCount);
		state.setCollisionCount(state.getCollisionCount() + numCollisions);
		state.setTurnCount(turn);
		
		if(displacementStep) {
			state.setDisplacementCount(displacementCount);
			state.setNextDisplacementAt(nextDisplacementStep);
			
			// KJ_DBG("Setting fwd", idx, tracingDirection);
			state.setForward(tracingDirection == 1);
			// KJ_DBG("Saving RNG state", idx);
			rng.save(state.mutateRngState());
			// KJ_DBG("Displacement step done", idx);
		}
		
		state.setStepSize(stepSize);
		
		++step;
	}
	
	// !!! The kernel returns by jumping to this label !!!
	THE_END:
	
	if(myData.getStopReason() == ::fsc::cu::FLTStopReason::EVENT_BUFFER_FULL) {
		if(step == state.getNumSteps()) {
			myData.setStopReason(::fsc::cu::FLTStopReason::COULD_NOT_STEP);
		}
	}
	
	// KJ_DBG("Kernel returned", (int) myData.getStopReason());
	
	// Copy state data back from local memory
	for(int i = 0; i < 3; ++i)
		statePos.set(i, x[i]);
	state.setNumSteps(step);
	state.setDistance(distance);
	state.setTheta(theta);
	
	// Note: The event count is not updated here but at the end of the loop
	// This ensures that events from unfinished steps do not get added
	
	// KJ_DBG("Kernel done");

	#undef FSC_FLT_RETURN
	#undef FSC_FLT_LOG_EVENT
}

}