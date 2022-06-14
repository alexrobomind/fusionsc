#include "kernels-flt.h"
#include "cudata.h"

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

struct KernelScheduler {
	constexpr static size_t STEPS_PER_ROUND = 1000;
	constexpr static size_t EVENTBUF_SIZE = 20;
	
	struct Round {
		Temporary<FLTKernelData> kernelData;
		Temporary<FLTKernelRequest> kernelRequest;
		
		Promise<void> whenDone;
		Vector<size_t> participants;
		
		size_t upperBound;
	};
	
	Round& prepareRound(size_t nParticipants) {
		Round round;
		
		auto data = round.kernelData.initData(nParticipants);
		for(size_t i = 0; i < nParticipants; ++i) {
			data[i].initState();
			auto events = data[i].initEvents(EVENTBUF_SIZE);
			
			for(event : events)
				event.initLocation(3);
		}
		
		return rounds.add(mv(round));
	}
	
	Round& prepareInitialRound(FLTKernelRequest::Reader requestTemplate, const TensorMap<Tensor<double, 2>> positions) {
		KJ_REQUIRE(positions.dimension(0) == 3);
		
		const size_t nParticipants = positions.dimension(1);
		Round& round = prepareRound(nParticipants);
		
		round.participants.addAll(kj::range<size_t>(0, nParticipants));	
		
		auto data = round.kernelData.getData();
		for(size_t i = 0; i < nParticipants; ++i) {
			auto state = data[i];
			
			auto pos = state.initPosition(3);
			for(unsigned char iDim = 0; iDim < 3; ++iDim)
				pos[iDim] = positions(iDim, i);
			
			state.setPhi0(std::atan2(pos[1], pos[0]));
		}
	}
	
	kj::Vector<Round> rounds;
};
	
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
}