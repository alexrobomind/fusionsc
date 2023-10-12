#include <catch2/catch_test_macros.hpp>

#include <fsc/devices/jtext.capnp.h>

#include "vmec.h"
#include "yaml.h"
#include "efit.h"
#include "magnetics.h"

using namespace fsc;

static inline kj::StringPtr vmecRequestYaml =
R"(
phiEdge: 1
freeBoundary:
  vacuumField:
    grid:
      rMin: 0.9
      rMax: 1.4
      zMin: -0.25
      zMax: 0.25
      
      nPhi: 3
      nR: 4
      nZ: 5
      
      mTor: 7
      
startingPoint:
  nTor: 2
  mPol: 1
  
  rCos:
    shape: [2, 3, 2]
    data: [
      0, 0,
      1, 0,
      0, 0,
      
      0, 0,
      1, 0,
      0.1, 0
    ]
  zSin:
    shape: [2, 3, 2]
    data: [
      0, 0,
      0, 0,
      0, 0,
      
      0, 0,
      0, 0,
      0.1, 0
    ]
massProfile:
  spline:
    type: akima
    locations: [0, 0.5, 1]
    values: [1, 0.5, 0]
iota:
  fromCurrent:
    totalCurrent: 0
    currentProfile:
      powerSeries: [1]
)"_kj;

static VmecRequest::Reader vmecRequest(VmecRequest::Builder req) {
    auto node = YAML::Load(vmecRequestYaml.cStr());
    load(req, node);
    
    auto field = req.getFreeBoundary().initVacuumFieldHl();
    parseGeqdsk(field.initAxisymmetricEquilibrium(), devices::jtext::EXAMPLE_GFILE.get());
    
    auto calculator = newFieldCalculator(kj::refcounted<CPUDevice>(CPUDevice::estimateNumThreads()));
    auto computeRequest = calculator.computeRequest();
    computeRequest.setField(field);
    computeRequest.setGrid(req.getFreeBoundary().getVacuumField().getGrid());
    
    req.getFreeBoundary().getVacuumField().setData(computeRequest.send().getComputedField().getData());
    return req.asReader();
}   

TEST_CASE("vmec-input") {
    auto l = newLibrary();
    auto lt = l -> newThread();
    auto& ws = lt -> waitScope();
	
	Temporary<VmecRequest> req;
	vmecRequest(req);
    
    KJ_DBG(generateVmecInput(req, kj::Path({"c:", "mgrid"})));
}