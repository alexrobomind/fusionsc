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

TEST_CASE("flm") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt -> waitScope();
	
	Temporary<LocalConfig> config;
	config.setPreferredDeviceType(ComputationDeviceType::CPU);
	auto req = createRoot(config).newMapperRequest();
	auto mapper = req.send().getService();
	
	auto mappingRequest = mapper.computeMappingRequest();
	
	Tensor<double, 2> startPoints(1, 3);
	startPoints(0, 0) = 1; startPoints(0, 1) = 0; startPoints(0, 2) = 0;
	writeTensor(startPoints, mappingRequest.getStartPoints());
	
	prepareToroidalField(mappingRequest.getField());
	mappingRequest.setNPhi(100);
	mappingRequest.setFilamentLength(10);
	mappingRequest.setCutoff(1);
	
	auto result = mappingRequest.send().wait(ws);
	// KJ_DBG(result);
	
	auto resultData = lt -> dataService().download(result.getMapping()).wait(ws);
	// KJ_DBG(resultData.get());
	
	auto fltReq = createRoot(config).newTracerRequest();
	auto flt = fltReq.send().getService();
	
	auto traceReq = flt.traceRequest();
	prepareToroidalField(traceReq.getField());
	
	traceReq.getStartPoints().setShape({3});
	traceReq.getStartPoints().setData({1.0, 0.0, 0.0});
	
	traceReq.setMapping(resultData);
	
	auto planes = traceReq.initPlanes(1);
	planes[0].getOrientation().setPhi(3.141592);
	
	traceReq.setTurnLimit(10000);
	// traceReq.setStepLimit(10000);
	traceReq.setStepSize(0.1);
	
	KJ_DBG("Sending request");
	auto response = traceReq.send().wait(ws);
	
	// KJ_DBG(response.getPoincareHits());
}

TEST_CASE("rflm") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt -> waitScope();
	
	Temporary<LocalConfig> config;
	config.setPreferredDeviceType(ComputationDeviceType::CPU);
	auto req = createRoot(config).newMapperRequest();
	auto mapper = req.send().getService();
	
	auto rflmRequest = mapper.computeRFLMRequest();
	//rflmRequest.setGridR({0.51, 1.0, 1.49});
	//rflmRequest.setGridZ({-0.49, -0.25, 0, 0.25, 0.49});
	rflmRequest.setGridR({1});
	rflmRequest.setGridZ({0});
	rflmRequest.setMappingPlanes({0});
	
	prepareToroidalField(rflmRequest.getField());
	
	auto mapping = rflmRequest.sendForPipeline().getMapping();
	auto data = lt->dataService().download(mapping).wait(ws);
	KJ_DBG(data.get());
}