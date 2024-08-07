#include <catch2/catch_test_macros.hpp>

#include <fsc/devices/jtext.capnp.h>

#include "vmec.h"
#include "structio.h"
#include "structio-yaml.h"
#include "efit.h"
#include "magnetics.h"

using namespace fsc;

static inline kj::StringPtr vmecRequestYaml =
R"(
phiEdge: 0.01
nTor: 1
mPol: 1

freeBoundary:
  vacuumField:
    grid:
      rMin: 0.9
      rMax: 1.4
      zMin: -0.25
      zMax: 0.25
      
      nPhi: 10
      nR: 10
      nZ: 10
      
      nSym: 5
      
startingPoint:
  nTor: 1
  mPol: 1
  toroidalSymmetry: 5
  
  rCos:
    shape: [2, 3, 2]
    data: [
      1, 0,
      0, 0,
      0, 0,
      
      1, 0.1,
      0, 0,
      0, 0
    ]
  zSin:
    shape: [2, 3, 2]
    data: [
      0, 0,
      0, 0,
      0, 0,
      
      0, 0.1,
      0, 0,
      0, 0
    ]
massProfile:
  spline:
    type: akima
    locations: [0, 0.3, 0.7, 1]
    values: [1, 0.7, 0.5, 0]
iota:
  fromCurrent:
    totalCurrent: 0
    currentProfile:
      powerSeries: [1]
)"_kj;

static void vmecRequest(VmecRequest::Builder req) {
	structio::load(vmecRequestYaml.asBytes(), *structio::createVisitor(req), structio::Dialect::YAML);
    
    auto field = req.getFreeBoundary().initVacuumFieldHl();
    parseGeqdsk(field.initAxisymmetricEquilibrium(), devices::jtext::EXAMPLE_GFILE.get());
    
    FieldCalculator::Client calculator = newFieldCalculator(CPUDevice::create(CPUDevice::estimateNumThreads()));
    auto computeRequest = calculator.computeRequest();
    computeRequest.setField(field);
    computeRequest.setGrid(req.getFreeBoundary().getVacuumField().getGrid());
    
    req.getFreeBoundary().getVacuumField().setData(computeRequest.send().getComputedField().getData());
}   

TEST_CASE("vmec-input") {
    auto l = newLibrary();
    auto lt = l -> newThread();
    auto& ws = lt -> waitScope();
	
	Temporary<VmecRequest> req;
	vmecRequest(req);
    
    KJ_DBG(generateVmecInput(req, kj::Path({"c:", "mgrid"})));
}

TEST_CASE("vmec-mgrid") {
    auto l = newLibrary();
    auto lt = l -> newThread();
    auto& ws = lt -> waitScope();
	
	Temporary<VmecRequest> req;
	vmecRequest(req);
	
	auto fs = kj::newDiskFilesystem();
	auto mgridPath = fs -> getCurrentPath().append("mgrid.nc");
	
	writeMGridFile(mgridPath, req.getFreeBoundary().getVacuumField()).wait(ws);
}

TEST_CASE("vmec-run", "[.]") {
    auto l = newLibrary();
    auto lt = l -> newThread();
    auto& ws = lt -> waitScope();
	
	VmecDriver::Client driver = createVmecDriver(LoopDevice::create(), newProcessScheduler("."));
	auto req = driver.runRequest();
	vmecRequest(req);
	
	auto resp = req.send().wait(ws);
	KJ_DBG(asYaml(resp));
	KJ_DBG(resp.getInputFile());
	KJ_DBG(resp.getStdout());
	KJ_DBG(resp.getStderr());
}

TEST_CASE("vmec-run-fb", "[.]") {
    auto l = newLibrary();
    auto lt = l -> newThread();
    auto& ws = lt -> waitScope();
	
	VmecDriver::Client driver = createVmecDriver(LoopDevice::create(), newProcessScheduler("."));
	auto req = driver.runRequest();
	vmecRequest(req);
	
	req.setFixedBoundary();
	
	auto resp = req.send().wait(ws);
	KJ_DBG(asYaml(resp));
	KJ_DBG(resp.getInputFile());
	KJ_DBG(resp.getStdout());
	KJ_DBG(resp.getStderr());
}

TEST_CASE("vmec-surf") {
    auto l = newLibrary();
    auto lt = l -> newThread();
    auto& ws = lt -> waitScope();
	
	VmecDriver::Client driver = createVmecDriver(LoopDevice::create(), newProcessScheduler("."));
	auto req = driver.computePositionsRequest();
	
	auto surf = req.initSurfaces();
	surf.setMPol(1);
	surf.setNTor(1);
	
	auto rcos = surf.getRCos();
	rcos.setShape({2, 3, 2});
	rcos.setData({
		1.0, 0.0,
		0.0, 0.0,
		0.0, 0.0,
		
		1.0, 0.1,
		0.0, 0.0,
		0.0, 0.0
	});
	
	auto zsin = surf.getZSin();
	zsin.setShape({2, 3, 2});
	zsin.setData({
		0.0, 0.0,
		0.0, 0.0,
		0.0, 0.0,
		
		0.0, 0.1,
		0.0, 0.0,
		0.0, 0.0
	});
	
	auto spt = req.getSPhiTheta();
	spt.setShape({3, 4});
	spt.setData({0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5 * fsc::pi, fsc::pi, 1.5 * fsc::pi});
	
	SECTION("no s") {
	}
	
	SECTION("custom s") {
		req.setSValues({0, 10});
	}
	
	auto result = req.send().wait(ws);
	KJ_DBG(result.getPhiZR());
}

TEST_CASE("vmec-surf-inv") {
    auto l = newLibrary();
    auto lt = l -> newThread();
    auto& ws = lt -> waitScope();
	
	VmecDriver::Client driver = createVmecDriver(LoopDevice::create(), newProcessScheduler("."));
	auto req = driver.invertPositionsRequest();
	
	auto surf = req.initSurfaces();
	surf.setMPol(1);
	surf.setNTor(1);
	
	auto rcos = surf.getRCos();
	rcos.setShape({2, 3, 2});
	rcos.setData({
		1.0, 0.0,
		0.0, 0.0,
		0.0, 0.0,
		
		1.0, 0.1,
		0.0, 0.0,
		0.0, 0.0
	});
	
	auto zsin = surf.getZSin();
	zsin.setShape({2, 3, 2});
	zsin.setData({
		0.0, 0.0,
		0.0, 0.0,
		0.0, 0.0,
		
		0.0, 0.1,
		0.0, 0.0,
		0.0, 0.0
	});
	
	auto pzr = req.getPhiZR();
	pzr.setShape({3, 4});
	pzr.setData({0, 0, 0, 0, 0, 0.05, 0, -0.05, 1.05, 1, 0.95, 1});
	
	SECTION("no s") {
	}
	
	SECTION("custom s") {
		req.setSValues({0, 10});
	}
	
	auto result = req.send().wait(ws);
	KJ_DBG(result.getSPhiTheta());
}
