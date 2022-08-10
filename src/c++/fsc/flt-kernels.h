#include <fsc/flt.capnp.cu.h>

#include "tensor.h"
#include "cudata.h"
#include "interpolation.h"
#include "geometry.h"
#include "random-kernels.h"

namespace fsc {
	
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
	inline EIGEN_DEVICE_FUNC void fltKernel(
		unsigned int idx,
		CuPtr<fsc::cu::FLTKernelData> pKernelData,
		TensorMap<Tensor<double, 4>> fieldData,
		CuPtr<fsc::cu::FLTKernelRequest> pRequest,
		CuPtr<const fsc::cu::MergedGeometry> pGeometry,
		CuPtr<const fsc::cu::IndexedGeometry> pGeometryIndex,
		CuPtr<const fsc::cu::IndexedGeometry::IndexData> pGeometryIndexData
	) {
		using Num = double;
		using V3 = Vec3<Num>;
		
		auto kernelData = *pKernelData;
		auto kernelRequest   = *pRequest;
		auto request = kernelRequest.getServiceRequest();
		
		auto parModel = request.getParallelModel();
		auto perpModel = request.getPerpendicularModel();
		
		auto geometry = *pGeometry;
		auto index = *pGeometryIndex;
		auto indexData = *pGeometryIndexData;
		
		// printf("Hello there\n");
		CUPNP_DBG("FLT kernel started", idx);
		
		// Extract local scratch space
		fsc::cu::FLTKernelData::Entry myData = kernelData.mutateData()[idx];
		fsc::cu::FLTKernelState state = myData.mutateState();
		auto statePos = state.mutatePosition();
		
		// Copy initial state data into local memory
		V3 x;
		for(int i = 0; i < 3; ++i)
			x[i] = statePos[i];
		
		uint32_t step = state.getNumSteps();
		uint32_t eventCount = state.getEventCount();
		Num distance = state.getDistance();
		double nextDisplacementStep = state.getNextDisplacementAt();
		uint32_t displacementCount = state.getDisplacementCount();
		MT19937 rng(state.getRngState());
		
		// Set up the magnetic field
		using InterpolationStrategy = LinearInterpolation<Num>;
		
		auto grid = request.getField().getGrid();
		SlabFieldInterpolator<InterpolationStrategy> interpolator(InterpolationStrategy(), grid);
		
		bool processDisplacements = perpModel.hasIsotropicDiffusionCoefficient();
		
		uint8_t tracingDirection = state.getForward() ? 1 : -1;
		
		auto rungeKuttaInput = [&](V3 x, Num t) -> V3 {
			V3 fieldValue = interpolator(fieldData, x);
			auto result = fieldValue / fieldValue.norm() * tracingDirection;
			
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
		#define FSC_FLT_LOG_EVENT(x) {\
			if(eventCount >= myData.getEvents().size() - 1) { \
				FSC_FLT_RETURN(EVENT_BUFFER_FULL); \
			} \
			\
			{\
				auto evt = currentEvent(); \
				evt.setStep(step); \
				evt.setDistance(distance); \
				\
				auto loc = evt.mutateLocation(); \
				for(int i = 0; i < 3; ++i) {\
					loc.set(i, x[i]); \
				}\
			}\
			\
			++eventCount; \
		}
		
		auto currentEvent = [&]() {
			return myData.mutateEvents()[eventCount];
		};
		
		// ... do the work ...
		
		while(true) {		
			// KJ_DBG("Beginning loop", step, distance);
			
			// Check various limits
			if(step >= request.getStepLimit() && request.getStepLimit() > 0)
				FSC_FLT_RETURN(STEP_LIMIT);
			
			if(distance >= request.getDistanceLimit() && request.getDistanceLimit() > 0)
				FSC_FLT_RETURN(DISTANCE_LIMIT);
			
			if(state.getTurnCount() >= request.getTurnLimit() && request.getTurnLimit() > 0)
				FSC_FLT_RETURN(TURN_LIMIT);
			
			if(x != x)
				FSC_FLT_RETURN(NAN_ENCOUNTERED);
			
			Num r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
			Num z = x[2];
			
			// KJ_DBG("In grid?", r, z, grid.getRMin(), grid.getRMax(), grid.getZMin(), grid.getZMax());
			
			if(r <= grid.getRMin() || r >= grid.getRMax() || z <= grid.getZMin() || z >= grid.getZMax())
				FSC_FLT_RETURN(OUT_OF_GRID);
			
			// KJ_DBG("Limits passed");
			
			V3 x2 = x;
			
			bool displacementStep = false;
			
			if(processDisplacements) {
				if(distance > nextDisplacementStep)
					displacementStep = true;
			}
			
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
				double isoDisplacement = std::sqrt(deltaT * perpModel.getIsotropicDiffusionCoefficient());
				// KJ_DBG(idx, isoDisplacement);
								
				for(int i = 0; i < 3; ++i) {
					x2[i] += isoDisplacement * normalDistributed[i];
				}
				
				++displacementCount;
			} else {
				// Regular tracing step
				kmath::runge_kutta_4_step(x2, .0, request.getStepSize(), rungeKuttaInput);
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
					
					currentEvent().mutatePhiPlaneIntersection().setPlaneNo(iPlane);
					FSC_FLT_LOG_EVENT(xCross);				
				}
			}
			
			// --- Check for new turns ---
						
			if(kmath::crossedPhi(phi1, phi2, phi0) && step > 1) {
				auto l = kmath::wrap(phi0 - phi1) / kmath::wrap(phi2 - phi1);
				V3 xCross = l * x2 + (1. - l) * x;
				
				// printf("New turn\n");
				
				// KJ_DBG(idx, state.getTurnCount());
				
				currentEvent().setNewTurn(state.getTurnCount() + 1);
				FSC_FLT_LOG_EVENT(xCross);		
				
				state.setTurnCount(state.getTurnCount() + 1);		
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
							
							auto loc1 = event1.mutateLocation();
							auto loc2 = event2.mutateLocation();
							
							for(int i = 0; i < 3; ++i) {
								double tmp = loc1[i];
								loc1.set(i, loc2[i]);
								loc2.set(i, tmp);
							}
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
							
							auto loc = evt.getLocation();
							for(int i = 0; i < 3; ++i)
								x[i] = loc[i];
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
			
			if(!displacementStep)
				distance += std::abs(request.getStepSize());
			else
				distance += (x2 - x).norm();
			
			x = x2;
			
			state.setEventCount(eventCount);
			state.setCollisionCount(state.getCollisionCount() + numCollisions);
			
			if(displacementStep) {
				state.setDisplacementCount(displacementCount);
				state.setNextDisplacementAt(nextDisplacementStep);
				
				// KJ_DBG("Setting fwd", idx, tracingDirection);
				state.setForward(tracingDirection == 1);
				// KJ_DBG("Saving RNG state", idx);
				rng.save(state.mutateRngState());
				// KJ_DBG("Displacement step done", idx);
			}
			
			++step;
		}
		
		// !!! The kernel returns by jumping to this label !!!
		THE_END:
		
		// KJ_DBG("Kernel returned", (int) myData.getStopReason());
		
		// Copy state data back from local memory
		for(int i = 0; i < 3; ++i)
			statePos.set(i, x[i]);
		state.setNumSteps(step);
		state.setDistance(distance);
		
		// Note: The event count is not updated here but at the end of the loop
		// This ensures that events from unfinished steps do not get added
		
		// KJ_DBG("Kernel done");
	
		#undef FSC_FLT_RETURN
		#undef FSC_FLT_LOG_EVENT
	}
}

REFERENCE_KERNEL(
	fsc::fltKernel,
	
	fsc::CuPtr<fsc::cu::FLTKernelData>,
	Eigen::TensorMap<Eigen::Tensor<double, 4>>,
	fsc::CuPtr<fsc::cu::FLTKernelRequest>,
	
	fsc::CuPtr<const fsc::cu::MergedGeometry>,
	fsc::CuPtr<const fsc::cu::IndexedGeometry>,
	fsc::CuPtr<const fsc::cu::IndexedGeometry::IndexData>
);