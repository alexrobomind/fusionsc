#include <catch2/catch_test_macros.hpp>

#include <kj/array.h>

#include <capnp/serialize-text.h>

#include "services.h"
#include "flt.h"
#include "local.h"
#include "magnetics.h"
#include "geometry.h"

#include <iostream>

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

TEST_CASE("flt") {
	auto l = newLibrary();
	auto lt = l->newThread();
	
	auto& ws = lt->waitScope();
	
	Temporary<LocalConfig> config;
	config.setPreferredDeviceType(ComputationDeviceType::CPU);
	
	RootService::Client root = createRoot(config);
	auto req = root.newTracerRequest();
	auto flt = req.send().getService();
	
	/*SECTION("basic-trace") {
		auto traceReq = flt.traceRequest();
		auto promise = traceReq.send().wait(ws);
	}*/
	
	SECTION("1start-trace") {	
		auto traceReq = flt.traceRequest();
		prepareToroidalField(traceReq.getField());
		
		// traceReq.getStartPoints().setShape({3, 4});
		// traceReq.getStartPoints().setData({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
		
		traceReq.getStartPoints().setShape({3});
		traceReq.getStartPoints().setData({1.0, 0.0, 0.0});
		
		auto planes = traceReq.initPlanes(1);
		planes[0].getOrientation().setPhi(3.141592);
		
		traceReq.setTurnLimit(10);
		// traceReq.setStepLimit(10000);
		traceReq.setStepSize(0.001);
		
		KJ_DBG("Sending request");
		auto response = traceReq.send().wait(ws);
		
		KJ_DBG("Done");
		//KJ_DBG(response); // This currently can't be printed due to a Capnp bug
	}
}

TEST_CASE("axis") {
	auto l = newLibrary();
	auto lt = l->newThread();
	
	auto& ws = lt->waitScope();
	
	Temporary<LocalConfig> config;
	config.setPreferredDeviceType(ComputationDeviceType::CPU);
	
	RootService::Client root = createRoot(config);
	auto req = root.newTracerRequest();
	auto flt = req.send().getService();
	
	auto axReq = flt.findAxisRequest();
	axReq.setStartPoint({1.0, 0.0, 0.0});
	prepareToroidalField(axReq.getField());
	axReq.setStepSize(0.01);
	auto response = axReq.send().wait(ws);
	
	Tensor<double, 2> data;
	readTensor(response.getAxis(), data);
	
	std::cout << "Mean " << data.mean(Eigen::array<int, 1>({0})) << std::endl;
}

#ifdef FSC_WITH_CUDA

TEST_CASE("flt-gpu") {
	auto l = newLibrary();
	auto lt = l->newThread();
	
	auto& ws = lt->waitScope();
	
	auto flt = newFLT(newGpuDevice());
	
	SECTION("1start-trace") {	
		auto traceReq = flt.traceRequest();
		prepareToroidalField(traceReq.getField());
		
		// traceReq.getStartPoints().setShape({3, 4});
		// traceReq.getStartPoints().setData({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
		
		traceReq.getStartPoints().setShape({3});
		traceReq.getStartPoints().setData({1.0, 0.0, 0.0});
		
		auto planes = traceReq.initPlanes(1);
		planes[0].getOrientation().setPhi(3.141592);
		
		traceReq.setTurnLimit(10);
		// traceReq.setStepLimit(10000);
		traceReq.setStepSize(0.001);
		
		KJ_DBG("Sending request");
		auto response = traceReq.send().wait(ws);
		
		KJ_DBG("Done");
		// KJ_DBG(response);
	}
}

#endif