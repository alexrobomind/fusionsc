#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <kj/array.h>

#include <capnp/serialize-text.h>

#include "services.h"
#include "flt.h"
#include "local.h"
#include "magnetics.h"
#include "geometry.h"
#include "textio.h"

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
		
		SECTION("simple") {}
		SECTION("adaptive") {
			auto adaptive = traceReq.getStepSizeControl().initAdaptive();
			adaptive.getErrorUnit().setIntegratedOver(10 * 2 * 3.141592);
			adaptive.setTargetError(0.001);
		}
		
		auto response = traceReq.send().wait(ws);
		
		// kj::VectorOutputStream os;
		// textio::save((capnp::DynamicStruct::Reader) response, os, textio::Dialect { textio::Dialect::YAML });
		
		// KJ_DBG(kj::heapString(os.getArray().asChars()));
		
		double x = response.getEndPoints().getData()[0];
		double y = response.getEndPoints().getData()[1];
		double z = response.getEndPoints().getData()[2];
		
		double r = sqrt(x*x + y*y);
		KJ_DBG(r, z);
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
	
	auto pos = response.getPos();
	REQUIRE(pos.size() == 3);
	
	using Catch::Matchers::WithinAbs;
	REQUIRE_THAT(pos[0], WithinAbs(1.0, 0.01));
	REQUIRE_THAT(pos[1], WithinAbs(0.0, 0.01));
	REQUIRE_THAT(pos[2], WithinAbs(0.0, 0.01));
}

TEST_CASE("calc-surf") {
	auto l = newLibrary();
	auto lt = l->newThread();
	
	auto& ws = lt->waitScope();
	
	Temporary<LocalConfig> config;
	config.setPreferredDeviceType(ComputationDeviceType::CPU);
	
	RootService::Client root = createRoot(config);
	auto req = root.newTracerRequest();
	auto flt = req.send().getService();
	
	// Compute field
	auto calculator = root.newFieldCalculatorRequest().send().getService();
	auto calcFieldReq = calculator.computeRequest();
	{
		auto grid = calcFieldReq.getGrid();
		grid.setRMin(4.0);
		grid.setRMax(6.0);
		grid.setZMin(-1.5);
		grid.setZMax(1.5);
		grid.setNPhi(1);
		grid.setNZ(60);
		grid.setNR(60);
		
		simpleTokamak(calcFieldReq.getField(), 5.5, 1.5, 25, -0.015);
	}
	
	auto calcFieldResponse = calcFieldReq.send().wait(ws);
	auto field = calcFieldResponse.getComputedField();
	
	// Find axis
	auto axReq = flt.findAxisRequest();
	{
		axReq.setField(field);
		axReq.setStartPoint({5.6, 0.0, 0.0});
		axReq.setStepSize(0.01);
		axReq.setNIterations(20);
		axReq.setNTurns(10);
	}
	
	auto response = axReq.send().wait(ws);
	KJ_DBG(response);
	
	auto iotaReq = flt.traceRequest();
	{
		iotaReq.setField(field);
		iotaReq.setTurnLimit(200);
		
		auto axis = response.getAxis().getData();
		auto nAxis = response.getAxis().getShape()[1];
		
		auto ci = iotaReq.getFieldLineAnalysis().initCalculateIota();
		ci.setUnwrapEvery(1);
		
		auto rAx = ci.initRAxis(nAxis);
		auto zAx = ci.initZAxis(nAxis);
		
		for(auto i : kj::range(0, nAxis)) {
			double x = axis[i];
			double y = axis[nAxis + i];
			double z = axis[2 * nAxis + i];
			double r = std::sqrt(x*x+y*y);
			
			KJ_DBG(r, z);
			
			zAx.set(i, z);
			rAx.set(i, r);
		}
		
		auto sp = iotaReq.getStartPoints();
		sp.setShape({3, 3});
		sp.setData({5.6, 5.7, 5.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
		
		iotaReq.getStepSizeControl().initAdaptive().setTargetError(1e-5);
	}
	
	auto iotaResponse = iotaReq.send().wait(ws);
	auto iotas = iotaResponse.getFieldLineAnalysis().getIotas();
	KJ_DBG(iotas);
	
	auto surfReq = flt.traceRequest();
	{
		surfReq.setField(field);
		surfReq.setTurnLimit(200);
		
		auto axis = response.getAxis().getData();
		auto nAxis = response.getAxis().getShape()[1];
		
		auto calSurf = surfReq.getFieldLineAnalysis().initCalculateFourierModes();
		calSurf.setIota(iotas);
		calSurf.setMMax(1);
		calSurf.setNMax(1);
		
		auto sp = surfReq.getStartPoints();
		sp.setShape({3, 3});
		sp.setData({5.6, 5.7, 5.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
		
		surfReq.getStepSizeControl().initAdaptive().setTargetError(1e-5);
	}
	
	auto surfResponse = surfReq.send().wait(ws);
	auto modes = surfResponse.getFieldLineAnalysis().getFourierModes();
	KJ_DBG(modes);
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
		
		auto response = traceReq.send().wait(ws);
		// KJ_DBG(response);
	}
}

#endif