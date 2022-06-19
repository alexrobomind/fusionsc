#include "kernels-flt.h"
#include "cudata.h"
#include "kernels.h"
#include "flt.h"

#include <algorithm>

#include <kj/vector.h>
// #include <capnp/serialize-text.h>
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
	constexpr static size_t EVENTBUF_SIZE = 100;
	
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
		
		round.participants.reserve(nParticipants);
		
		return rounds.add(mv(round));
	}
	
	Round& setupInitialRound() {
		KJ_REQUIRE(positions.dimension(0) == 3);
		
		KJ_REQUIRE(rounds.size() == 0, "Internal error");
		
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
		
		round.kernelRequest = request.asReader();
		
		return round;
	}
	
	Round& setupFollowupRound() {
		KJ_REQUIRE(rounds.size() > 0, "Internal error");
		
		// Check previous round
		Round& prevRound = rounds[rounds.size() - 1];
		
		// Count unfinished participants
		kj::Vector<size_t> unfinished;
		auto kDataIn = prevRound.kernelData.getData();
		
		for(size_t i = 0; i < kDataIn.size(); ++i) {
			if(!isFinished(kDataIn[i])) unfinished.add(i);
		}
		
		KJ_REQUIRE(unfinished.size() > 0, "Internal error");
		
		Round& newRound = prepareRound(unfinished.size());
		auto kDataOut = newRound.kernelData.getData();
		for(size_t i = 0; i < unfinished.size(); ++i) {
			newRound.participants.add(unfinished[i]);
			
			auto entryOut = kDataOut[i];
			auto entryIn  = kDataIn[unfinished[i]];
			
			entryOut.setState(entryIn.getState());
			entryOut.getState().setEventCount(0);
		}
		
		newRound.kernelRequest = request.asReader();
		
		return newRound;
	}
	
	Round& setupRound() {
		if(rounds.size() == 0)
			return setupInitialRound();
		else
			return setupFollowupRound();
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
	
	Promise<void> runRound() {
		auto& round = setupRound();
		return startRound(round);
	}		
	
	Promise<void> run() {
		if(rounds.size() == 0) {
			return runRound().then([this]() { return run(); });
		}
			
		auto& round = rounds[rounds.size() - 1];
			
		if(isFinished(round))
			return READY_NOW;
		
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
				auto kData = rounds[i].kernelData.getData()[*pIdx];
				size_t eventCount = kData.getState().getEventCount();
				
				nEvents += eventCount;
			}
		}
		
		Temporary<FLTKernelData::Entry> result;
		auto eventsOut = result.initEvents(nEvents);
		
		size_t iEvent = 0;
		for(size_t i = 0; i < nRounds; ++i) {
			KJ_IF_MAYBE(pIdx, participantIndices[i]) {
				auto kData = rounds[i].kernelData.getData()[*pIdx];
				size_t eventCount = kData.getState().getEventCount();
				
				auto eventsIn = kData.getEvents();
				KJ_ASSERT(eventCount <= eventsIn.size());
				
				for(size_t iEvtIn = 0; iEvtIn < eventCount; ++iEvtIn)
					eventsOut.setWithCaveats(iEvent++, eventsIn[iEvtIn]);
				
				result.setStopReason(kData.getStopReason());
				result.setState(kData.getState());
			}
		}
		
		return result;
	}
};

template<typename Device>
struct FLTImpl : public FLT::Server {
	Own<Device> device;
	LibraryThread lt;
	
	FLTImpl(LibraryThread& lt, Own<Device> device) : device(mv(device)), lt(lt->addRef()) {}
	
	Promise<void> trace(TraceContext ctx) override {
		KJ_DBG("Processing trace request");
		ctx.allowCancellation();
		auto request = ctx.getParams();
		
		return lt->dataService().download(request.getField().getData())
		.then([ctx, request, this](LocalDataRef<Float64Tensor> fieldData) mutable {
			KJ_DBG("Converting request");
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
			KJ_REQUIRE(startPointShape.size() >= 1, "Start points must have at least 1 dimension");
			KJ_REQUIRE(startPointShape[0] == 3, "First dimension of start points must have size 3");
			
			int64_t nStartPoints = 1;
			for(size_t i = 1; i < startPointShape.size(); ++i)
				nStartPoints *= startPointShape[i];
			
			Temporary<Float64Tensor> reshapedStartPoints;
			reshapedStartPoints.setData(inStartPoints.getData());
			{
				reshapedStartPoints.setShape({3, (uint64_t) nStartPoints});
				// shape[0] = 3; shape[1] = nStartPoints;
			}			
			
			Tensor<double, 2> positions = mapTensor<Tensor<double, 2>>(reshapedStartPoints.asReader())
				-> shuffle(Eigen::array<int, 2>{1, 0});
			
			auto calc = heapHeld<TraceCalculation<Device>>(
				*device, mv(kernelRequest), mv(field), mv(positions)
			);
			
			KJ_DBG("Running");
			return calc->run()
			.then([ctx, calc, request, startPointShape, nStartPoints]() mutable {
				KJ_DBG("Extracting response");
				int64_t nTurns = 0;
				
				auto resultBuilder = kj::heapArrayBuilder<Temporary<FLTKernelData::Entry>>(nStartPoints);
				for(size_t i = 0; i < nStartPoints; ++i)
					resultBuilder.add(calc->consolidateRuns(i));
				
				auto kData = resultBuilder.finish();
								
				for(auto& entry : kData) {
					nTurns = std::max(nTurns, (int64_t) entry.getState().getTurnCount());
				}
				
				if(request.getTurnLimit() > 0)
					nTurns = std::min(nTurns, (int64_t) request.getTurnLimit());
				
				int64_t nSurfs = request.getPoincarePlanes().size();
				
				Tensor<double, 4> pcCuts(nTurns, nStartPoints, nSurfs, 3);
				for(int64_t iStartPoint = 0; iStartPoint < nStartPoints; ++iStartPoint) {
					auto entry = kData[iStartPoint].asReader();
					auto state = entry.getState();
					auto events = entry.getEvents();
								
					int64_t iTurn = 0;
					
					for(auto evt : events) {						
						KJ_REQUIRE(!evt.isNotSet(), "Internal error");
												
						if(evt.isNewTurn()) {
							iTurn = evt.getNewTurn();
						} else if(evt.isPhiPlaneIntersection()) {
							auto ppi = evt.getPhiPlaneIntersection();
							
							auto loc = evt.getLocation();
							for(int64_t iDim = 0; iDim < 3; ++iDim) {
								pcCuts(iTurn, iStartPoint, ppi.getPlaneNo(), iDim) = loc[iDim];
							}
						}
					}
				}
				
				auto results = ctx.getResults();
				results.setNTurns(nTurns);
				writeTensor(pcCuts, results.getPoincareHits());
				
			}).attach(cp(fieldData)).attach(calc.x());
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
	
	// TODO: Make this accept data service instead
	FLT::Client newCpuTracer(LibraryThread& lt) {
		return kj::heap<FLTImpl<Eigen::ThreadPoolDevice>>(lt, newThreadPoolDevice());
	}
}