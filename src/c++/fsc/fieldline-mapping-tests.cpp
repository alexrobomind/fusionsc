#include <catch2/catch_test_macros.hpp>

#include <fsc/geometry-test.capnp.h>
#include <fsc/magnetics-test.capnp.h>

#include "data.h"
#include "services.h"
#include "tensor.h"

using namespace fsc;

namespace {

void prepareToroidalField(ComputedField::Builder field) {
	auto grid = field.initGrid();
	grid.setNR(2);
	grid.setNZ(2);
	grid.setNPhi(1);
	
	grid.setRMin(0.5);
	grid.setRMax(1.5);
	grid.setZMin(-0.5);
	grid.setZMax(1.5);
	
	Temporary<Float64Tensor> fieldData;
	fieldData.setShape({1, 2, 2, 3});
	{
		auto data = fieldData.initData(12);
		for(size_t i = 0; i < 12; ++i)
			data.set(i, i % 3 == 0 ? 1 : 0);
	}
	
	field.setData(getActiveThread().dataService().publish(fieldData.asReader()));
}

}

TEST_CASE("rflm") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt -> waitScope();
	
	Temporary<LocalConfig> config;
	config.setPreferredDeviceType(ComputationDeviceType::CPU);
	
	RootService::Client root = createRoot(config);
	auto req = root.newMapperRequest();
	auto mapper = req.send().getService();
	
	auto rflmRequest = mapper.computeRFLMRequest();
	rflmRequest.setGridR({0.51, 1.0, 1.49});
	rflmRequest.setGridZ({-0.49, -0.25, 0, 0.25, 0.49});
	rflmRequest.setMappingPlanes({0});
	
	prepareToroidalField(rflmRequest.getField());
	
	auto mapping = rflmRequest.sendForPipeline().getMapping();
	auto data = lt->dataService().download(mapping).wait(ws);
	
	auto fltReq = root.newTracerRequest();
	auto flt = fltReq.send().getService();
	
	auto traceReq = flt.traceRequest();
	prepareToroidalField(traceReq.getField());
	
	traceReq.getStartPoints().setShape({3});
	traceReq.getStartPoints().setData({1.0, 0.0, 0.0});
	
	traceReq.setMapping(mapping);
	
	auto planes = traceReq.initPlanes(1);
	planes[0].getOrientation().setPhi(3.141592);
	
	traceReq.setTurnLimit(1000);
	traceReq.setDistanceLimit(0);
	traceReq.setStepSize(0.1);
	
	auto response = traceReq.send().wait(ws);
}