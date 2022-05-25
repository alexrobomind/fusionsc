#include <fsc/flt.capnp.cu.h>

#include "tensor.h"

namespace fsc {
	
	namespace kmath {
		template<typename Num>
		Num wrap(Num x) { 
			x %= 2 * pi;
			x += pi;
			x %= 2 * pi;
			x -= pi;
			return x;
		};
		
			// Check if the [phi1, phi2) interval crosses across phi0.
		template<typename Num>	
		bool crossedPhi(Num phi1, Num phi2, Num phi0) {
			auto d1 = wrap(phi0 - phi1);
			auto d2 = wrap(phi2 - phi1);
			
			if(d1 >= 0 && d2 > 0 && d1 < d2)
				return true;
			
			if(d1 <= 0 && d2 < 0 && d1 > d2)
				return true;
			
			return false;
		};
	}

	
	/*
	
	Some special programming rules when working inside this function:
	
	- The function uses a goto-block for termination which is equivalent to a try-finally construct.
	  Therefore, NEVER use "return" to leave. Always use the FSC_FLT_RETURN macro, which will also
	  carry the termination reason.
	- The fieldline tracer uses a log buffer to record special events such as Poincare intersections,
	  direction reversals, or collisions.
	
	*/
	
	EIGEN_DEVICE_FUNC fltKernel(
		unsigned int idx,
		fsc::cu::FLTKernelData kernelData
		const fsc::cu::FLTKernelRequest request
	) {
		using Num = float;
		using V3 = Vec3<Num>;
		
		// Extract local scratch space
		fsc::cu::FLTKernelData::Entry myData = kernelData.mutateData()[idx];
		fsc::cu::FLTKernelState state = myData.mutateState();
		auto statePos = state.mutatePosition();
		
		// The kernel terminates its execution with this macro
		#define FSC_FLT_RETURN(reason) \
			state.setStopReason(FLTStopReason::reason); \
			goto THE_END
		
		// Event logging
		// As this involves our return macro, it can't go
		// into a lambda
		#define FSC_FLT_LOG_EVENT(x) \
			{\
				auto evt = currentEvent(); \
				evt.setStep(step); \
				\
				auto loc = evt.mutateLocation() \
				for(int i = 0; i < 3; ++i) \
					loc.set(i, x[i]); \
			}\
			\
			++eventCount; \
			\
			if(eventCount >= myData.getEvents().size()) { \
				FSC_FLT_RETURN(EVENT_BUFFER_FULL); \
			} \
		
		auto currentEvent = [&]() {
			return myData.mutateEvents()[eventCount];
		}
		
		// Copy initial state data into local memory
		V3 x;
		for(int i = 0; i < 3; ++i)
			x[i] = statePos[i];
		
		uint32_t step = state.getStep();
		uint32_t eventCount = state.getEventCount();
		double distance = state.getDistance();
		
		// ... do the work ...
		
		while(true) {			
			// Check various limits
			if(step >= request.getStepLimit() && request.getStepLimit() > 0)
				FSC_FLT_RETURN(STEP_LIMIT);
			
			if(distance >= request.getDistanceLimit() && request.getDistanceLimit() > 0)
				FSC_FLT_RETURN(DISTANCE_LIMIT);
			
			if(state.getTurn() >= request.getTurnLimit() && request.getTurnLimit() > 0)
				FSC_FLT_RETURN(TURN_LIMIT);
			
			// TODO: Add runge kutta step here
			V3 x2;
			KJ_UNIMPLEMENTED("Runge Kutta");
			
			// --- Check for phi crossings ---
			
			Num phi1 = atan2(x[1], x[0]);
			Num phi2 = atan2(x2[1], x2[0]);
			
			Num phi0 = state.getPhi0();
			if(kmath::crossedPhi(phi1, phi2, phi0)) {
				auto l = kmath::wrap(phi0 - phi1) / kmath::wrap(phi2 - phi1);
				V3 xCross = l * x2 + (1. - l) * x1;
				
				state.setTurn(state.getTurn() + 1);
				
				currentEvent().setNewTurn(state.getTurn());
				FSC_FLT_LOG_EVENT(xCross);				
			}
			
			// --- Advance the step after all events are processed ---
			
			x = x2;
			distance += request.getStepSize();
			++step;
			state.getEventCount(eventCount);
		}
		
		// !!! The kernel returns by jumping to this label !!!
		THE_END:
		
		// Copy state data back from local memory
		for(int i = 0; i < 3; ++i)
			statePos.set(i, x[i]);
		state.setStep(step):
		state.setDistance(distance);
		
		// Note: The event count is not updated here but at the end of the loop
		// This ensures that events from unfinished steps do not get added
	
		#undef FSC_FLT_RETURN
		#undef FSC_FLT_LOG_EVENT
	}
}