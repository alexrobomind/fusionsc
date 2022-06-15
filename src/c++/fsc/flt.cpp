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
	
	TraceCalculation(Temporary<FLTKernelRequest>&& newRequest, Tensor< newFieldData) :
		request(mv(newRequest)),
		mappedRequest(request),
		fieldData(mv(newFieldData)),
		mappedFieldData(fieldData)
	{
		mappedRequest.copyToDevice();
		mappedFieldData.copyToDevice();
	}
		
	
	struct Round {
		Temporary<FLTKernelData> kernelData;
		Temporary<FLTKernelRequest> kernelRequest;
		
		kj::Vector<size_t> participants;
		
		size_t upperBound;
	};
	
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
	
	Round& setupRound() {
		if(rounds.size() == 0) {
			return setupInitialRound();
		}
		KJ_UNIMPLEMENTED("Multiple rounds not supported");
	}
	
	Tensor<double, 2> positions;
	
	LocalDataRef<Float64Tensor> fieldData;
	Maybe<
	MappedMessage<cu::Float64Tensor> mappedFieldData;
	
	Temporary<FLTKernelRequest> request;
	
	kj::Vector<Round> rounds;
};

template<typename Device>
struct FLTImpl : public FLT::Server {
	Own<Device> device;
	
	FLTImpl(Own<Device> device) : device(mv(device)) {}
	
	Promise<void> trace(TraceContext ctx) override {
		return READY_NOW;
	}
};

template struct TraceCalculation<Eigen::ThreadPoolDevice>;
	
}

namespace fsc {
	void calc(Eigen::DefaultDevice& d) {
		capnp::MallocMessageBuilder mb;
		CupnpMessage<cu::FLTKernelData> kernelData(mb);
		CupnpMessage<cu::Float32Tensor> f32tensor(mb);
		CupnpMessage<cu::FLTKernelRequest> kernelRequest(mb);
		auto calculation = FSC_LAUNCH_KERNEL(
			fltKernel,
			
			d, 3,
			
			kernelData, f32tensor, kernelRequest
		);
	}
	
	FLT::Client newCpuTracer() {
		return kj::heap<FLTImpl<Eigen::ThreadPoolDevice>>(newThreadPoolDevice());
	}
}