#include <catch2/catch_test_macros.hpp>

#include <fsc/magnetics-test.capnp.h>
#include <fsc/magnetics-test.capnp.cu.h>

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
	
	TVec3d ref;
	ref(0) = 1;
	ref(1) = 0;
	ref(2) = 0;
	
	for(size_t iPhi = 0; iPhi < grid.getNPhi(); ++iPhi) {
	for(size_t iR = 0; iR < grid.getNR(); ++iR) {
		double r = grid.getRMin() + (grid.getRMax() - grid.getRMin()) / (grid.getNR() - 1) * iR;
		double reference = 2e-7 / r * sin(atan2(1, r));
		TVec3d upscaled = zPlane.chip(iR, 2).chip(iPhi, 1) / reference;
		TensorFixedSize<double, Eigen::Sizes<>> dist = (upscaled - ref).square().sum().sqrt();
		
		KJ_REQUIRE(dist() < 0.01);
	}}
}

#ifdef FSC_WITH_CUDA

TEST_CASE("build-field-gpu") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
		
	auto calc = newGPUFieldCalculator(lt);
	
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
	
	EVec3d ref;
	ref(0) = 1;
	ref(1) = 0;
	ref(2) = 0;
	
	for(size_t iPhi = 0; iPhi < grid.getNPhi(); ++iPhi) {
	for(size_t iR = 0; iR < grid.getNR(); ++iR) {
		double r = grid.getRMin() + (grid.getRMax() - grid.getRMin()) / (grid.getNR() - 1) * iR;
		double reference = 2e-7 / r * sin(atan2(1, r));
		EVec3d upscaled = zPlane.chip(iR, 2).chip(iPhi, 1) / reference;
		TensorFixedSize<double, Eigen::Sizes<>> dist = (upscaled - ref).square().sum().sqrt();
		
		KJ_REQUIRE(dist() < 0.01);
	}}
	
}

#endif

}