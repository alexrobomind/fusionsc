#include "kernels-flt.h"
#include "cudata.h"
#include "kernels.h"
#include "flt.h"

#include <kj/vector.h>

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

template<typename Device>
struct TraceCalculation {
	constexpr static size_t STEPS_PER_ROUND = 1000;
	constexpr static size_t EVENTBUF_SIZE = 20;
	
	template<typename T>
	using MappedMessage = MapToDevice<CupnpMessage<T>, Device>;
		
	struct Round {
		Temporary<FLTKernelData> kernelData;
		Temporary<FLTKernelRequest> kernelRequest;
		
		kj::Vector<size_t> participants;
		
		size_t upperBound;
	};
	
	Device& device;
	Tensor<double, 2> positions;
	
	Own<TensorMap<const Tensor<double, 4>>> field;
	MapToDevice<TensorMap<const Tensor<double, 4>>, Device> deviceField;
	
	Temporary<FLTKernelRequest> request;
	kj::Vector<Round> rounds;
	
	TraceCalculation(Device& device, Temporary<FLTKernelRequest>&& newRequest, Own<TensorMap<const Tensor<double, 4>>> newField, Tensor<double, 2> positions) :
		device(device),
		positions(mv(positions)),
		
		field(mv(newField)),
		deviceField(mapToDevice(*field, device)),
		
		request(mv(newRequest))
	{
		deviceField.updateDevice();
	}
	
	// Prepares the memory structure for a round
	Round& prepareRound(size_t nParticipants) {
		Round round;
		
		auto data = round.kernelData.initData(nParticipants);
		for(size_t i = 0; i < nParticipants; ++i) {
			data[i].initState();
			auto events = data[i].initEvents(EVENTBUF_SIZE);
			
			for(auto event : events)
				event.initLocation(3);
		}
		
		return rounds.add(mv(round));
	}
	
	Round& setupInitialRound() {
		KJ_REQUIRE(positions.dimension(0) == 3);
		
		const size_t nParticipants = positions.dimension(1);
		Round& round = prepareRound(nParticipants);
		
		round.participants.addAll(kj::range<size_t>(0, nParticipants));	
		
		auto data = round.kernelData.getData();
		for(size_t i = 0; i < nParticipants; ++i) {
			auto state = data[i].initState();
			
			auto pos = state.initPosition(3);
			for(unsigned char iDim = 0; iDim < 3; ++iDim)
				pos.set(iDim, positions(iDim, i));
			
			state.setPhi0(std::atan2(pos[1], pos[0]));
		}
		
		return round;
	}
	
	Promise<void> startRound(Round& r) {
		CupnpMessage<cu::FLTKernelData> kernelData(r.kernelData);
		CupnpMessage<cu::FLTKernelRequest> kernelRequest(r.kernelRequest);
		
		return FSC_LAUNCH_KERNEL(
			fltKernel, device,
			r.kernelData.getData().size(),
			
			kernelData, FSC_KARG(deviceField, IN), kernelRequest
		).then([this]() {
			return hostMemSynchronize(device);
		});
	}
	
	bool isFinished(FLTKernelData::Entry::Reader entry) {
		auto stopReason = entry.getStopReason();
		
		if(stopReason == FLTStopReason::UNKNOWN) {
			KJ_FAIL_REQUIRE("Kernel stopped for unknown reason");
		}
		
		if(stopReason == FLTStopReason::EVENT_BUFFER_FULL)
			return false;
		
		if(stopReason == FLTStopReason::STEP_LIMIT && entry.getState().getNumSteps() < request.getStepLimit())
			return false;
		
		return true;
	}
	
	bool isFinished(Round& r) {
		for(auto kDatum : r.kernelData.getData()) {
			if(!isFinished(kDatum))
				return false;
		}
		
		return true;
	}
	
	Promise<void> run() {
		auto& round1 = setupInitialRound();
		Promise<void> calculation = startRound(round1);
		
		//TODO: Followup rounds
		
		return calculation.then([&]() {
			KJ_REQUIRE(isFinished(round1), "FLT can not handle multi-round launches yet");
		});
	}
};

template<typename Device>
struct FLTImpl : public FLT::Server {
	Own<Device> device;
	LibraryThread lt;
	
	FLTImpl(Own<Device> device) : device(mv(device)) {}
	
	Promise<void> trace(TraceContext ctx) override {
		ctx.allowCancellation();
		auto request = ctx.getParams();
		
		return lt->dataService().download(request.getField().getData())
		.then([ctx, request, this](LocalDataRef<Float64Tensor> fieldData) mutable {
			// Extract kernel request
			Temporary<FLTKernelRequest> kernelRequest;
			kernelRequest.setPhiPlanes(request.getPoincarePlanes());
			kernelRequest.setTurnLimit(request.getTurnLimit());
			kernelRequest.setDistanceLimit(request.getDistanceLimit());
			kernelRequest.setStepLimit(request.getStepLimit());
			kernelRequest.setStepSize(request.getStepSize());
			kernelRequest.setGrid(request.getField().getGrid());
			
			// Extract field data
			auto field = mapTensor<Tensor<double, 4>>(fieldData.get());
			
			// Extract positions
			auto inStartPoints = request.getStartPoints();
			auto startPointShape = inStartPoints.getShape();
			KJ_REQUIRE(startPointShape.size() >= 2, "Start points must have at least 1 dimension");
			KJ_REQUIRE(startPointShape[0] == 3, "First dimension of start points must have size 3");
			
			size_t nStartPoints = 1;
			for(size_t i = 1; i < startPointShape.size(); ++i)
				nStartPoints *= startPointShape[i];
			
			Temporary<Float64Tensor> reshapedStartPoints;
			reshapedStartPoints.setData(inStartPoints.getData());
			{
				reshapedStartPoints.setShape({3, nStartPoints});
				// shape[0] = 3; shape[1] = nStartPoints;
			}			
			
			Tensor<double, 2> positions = mapTensor<Tensor<double, 2>>(reshapedStartPoints.asReader())
				-> shuffle(Eigen::array<int, 2>{1, 0});
			
			auto calc = heapHeld<TraceCalculation<Device>>(
				*device, mv(kernelRequest), mv(field), mv(positions)
			);
			
			return calc->run()
			.then([ctx, calc, request, startPointShape, nStartPoints]() mutable {
				// Count maximum number of turns
				KJ_REQUIRE(calc->rounds.size() == 1, "Only supports single-round execution");
				
				auto& round = calc->rounds[0];
				size_t nTurns = 0;
				
				auto kData = round.kernelData.getData();
				
				// Single-round strategy
				KJ_REQUIRE(kData.size() == nStartPoints, "Internal error");
				
				for(auto entry : kData) {
					nTurns = std::max(nTurns, (size_t) entry.getState().getTurnCount());
				}
				
				nTurns = std::min(nTurns, (size_t) request.getTurnLimit());
				size_t nSurfs = request.getPoincarePlanes().size();
				
				Tensor<double, 4> pcCuts(nTurns, nStartPoints, nSurfs, 3);
				for(size_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
					auto entry = kData[iStartPoint];
					auto state = entry.getState();
					auto events = entry.getEvents();
			
					// DO NOT iterate over the whole event list
					// There might be invalid events at the end that
					// got rolled back
					size_t nEvents = state.getEventCount();
					KJ_REQUIRE(nEvents <= events.size(), "Internal error");
					
					size_t iTurn = 0;
					
					for(size_t iEvt = 0; iEvt < nEvents; ++iEvt) {
						auto evt = events[iEvt];
						
						KJ_REQUIRE(!evt.isNotSet(), "Internal error");
						
						if(evt.isNewTurn()) {
							iTurn = evt.getNewTurn();
							KJ_DBG(iTurn);
						} else if(evt.isPhiPlaneIntersection()) {
							auto ppi = evt.getPhiPlaneIntersection();
							KJ_DBG(ppi);
							
							auto loc = evt.getLocation();
							for(size_t iDim = 0; iDim < 3; ++iDim) {
								pcCuts(iTurn, iStartPoint, ppi.getPlaneNo(), iDim) = loc[iDim];
							}
						}
					}
				}
			}).attach(calc.x());
		}).attach(thisCap());
	}
};

template struct TraceCalculation<Eigen::ThreadPoolDevice>;
	
}

namespace fsc {
	void calc(Eigen::DefaultDevice& d) {
		capnp::MallocMessageBuilder mb;
		CupnpMessage<cu::FLTKernelData> kernelData(mb);
		Tensor<double, 4> field(3, 1, 1, 1);
		CupnpMessage<cu::FLTKernelRequest> kernelRequest(mb);
		auto calculation = FSC_LAUNCH_KERNEL(
			fltKernel,
			
			d, 3,
			
			kernelData, field, kernelRequest
		);
		
		Temporary<Float64Tensor> testTensor;
		mapTensor<const Tensor<double, 3>>(testTensor.asReader());
	}
	
	FLT::Client newCpuTracer() {
		return kj::heap<FLTImpl<Eigen::ThreadPoolDevice>>(newThreadPoolDevice());
	}
}