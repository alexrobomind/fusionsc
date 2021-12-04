#include <catch2/catch_test_macros.hpp>

#include <fsc/magnetics-test.capnp.h>

#include "magnetics.h"
#include "local.h"
#include "data.h"
#include "tensor.h"

namespace fsc {

TEST_CASE("build-field") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
		
	auto calc = newCPUFieldCalculator(lt);
	
	auto grid = TEST_GRID.get();
	auto sessRequest = calc.getRequest();
	sessRequest.setGrid(grid);
	
	auto session = sessRequest.send().wait(ws).getSession();
	
	auto computeRequest = session.computeRequest();
	computeRequest.setField(WIRE_FIELD.get());
	
	Temporary<ComputedField> computed(
		computeRequest.send().wait(ws).getComputedField()
	);
	
	LocalDataRef<Float64Tensor> data = lt->dataService().download(computed.getData()).wait(ws);
	auto fieldOut = readTensor<Tensor<double, 4>>(data.get());
	
	// col major axes 3, nPhi, nZ, R
	auto zPlane = fieldOut.chip(grid.getNZ() / 2, 2);
	
	for(size_t iPhi = 0; iPhi < grid.getNPhi(); ++iPhi) {
	for(size_t iR = 0; iR < grid.getNR(); ++iR) {
		double r = grid.getRMin() + (grid.getRMax() - grid.getRMin()) / (grid.getNR() - 1) * iR;
		double reference = 2e-7 / r * sin(atan2(1, r));
		Vec3d upscaled = zPlane.chip(iR, 2).chip(iPhi, 1) / reference;
		KJ_LOG(WARNING, upscaled(0), upscaled(1), upscaled(2));
	}}
}

}