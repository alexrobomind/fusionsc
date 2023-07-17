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
		
	auto grid = TEST_GRID.get();
	auto session = newFieldCalculator(kj::refcounted<CPUDevice>(CPUDevice::estimateNumThreads()));
	
	auto computeRequest = session.computeRequest();
	computeRequest.setField(WIRE_FIELD.get());
	computeRequest.setGrid(grid);
	
	KJ_DBG(grid);
		
	Temporary<ComputedField> computed(
		computeRequest.send().wait(ws).getComputedField()
	);
	
	LocalDataRef<Float64Tensor> data = lt->dataService().download(computed.getData()).wait(ws);
	auto fieldOut = readTensor<Tensor<double, 4>>(data.get());
	
	// col major axes 3, nR, nZ, nPhi
	auto zPlane = fieldOut.chip(grid.getNZ() / 2, 2);
	
	TVec3d ref;
	ref(0) = 1;
	ref(1) = 0;
	ref(2) = 0;
	
	for(size_t iPhi = 0; iPhi < grid.getNPhi(); ++iPhi) {
	for(size_t iR = 0; iR < grid.getNR(); ++iR) {
		double r = grid.getRMin() + (grid.getRMax() - grid.getRMin()) / (grid.getNR() - 1) * iR;
		double reference = 2e-7 / r * sin(atan2(1, r));
		TVec3d upscaled = zPlane.chip(iPhi, 2).chip(iR, 1) / reference;
		TensorFixedSize<double, Eigen::Sizes<>> dist = (upscaled - ref).square().sum().sqrt();
		// KJ_DBG(dist());
		
		KJ_REQUIRE(dist() < 0.01);
	}}
	
	// Check field evaluation
	
	auto evalRequest = session.evaluateXyzRequest();
	evalRequest.setField(computed);
	
	uint32_t nPhi = 20;
	uint32_t nR = 100;
	evalRequest.getPoints().setShape({3, nPhi, nR});
	auto reqData = evalRequest.getPoints().initData(3 * nPhi * nR);
	
	for(auto iPhi : kj::range(0, nPhi)) {
		for(auto iR : kj::range(0, nR)) {
			double r = grid.getRMin() + (grid.getRMax() - grid.getRMin()) / (nR - 1) * iR;
			double phi = 2 * fsc::pi / nPhi * iPhi;
			
			reqData.set(iR + nR * iPhi + 0 * nR * nPhi, r * std::cos(phi));
			reqData.set(iR + nR * iPhi + 1 * nR * nPhi, r * std::sin(phi));
			reqData.set(iR + nR * iPhi + 2 * nR * nPhi, 0);
		}
	}
	
	auto evalResult = evalRequest.send().wait(ws);
	auto vals = evalResult.getValues().getData();
	
	for(auto iPhi : kj::range(0, nPhi)) {
		for(auto iR : kj::range(0, nR)) {
			double bx = vals[iR + nR * iPhi + 0 * nR * nPhi];
			double by = vals[iR + nR * iPhi + 1 * nR * nPhi];
			double bz = vals[iR + nR * iPhi + 2 * nR * nPhi];
			
			double r = grid.getRMin() + (grid.getRMax() - grid.getRMin()) / (nR - 1) * iR;
			double phi = 2 * fsc::pi / nPhi * iPhi;
			
			double ref = 2e-7 / r * sin(atan2(1, r));
			double refX = -std::sin(phi) * ref;
			double refY =  std::cos(phi) * ref;
			
			KJ_REQUIRE(std::abs(bx / ref - refX / ref) < 0.05);
			KJ_REQUIRE(std::abs(by / ref - refY / ref) < 0.05);
			KJ_REQUIRE(std::abs(bz / ref) < 0.05);
		}
	}			
}

#ifdef FSC_WITH_CUDA

TEST_CASE("build-field-gpu") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	auto grid = TEST_GRID.get();
	auto session = newFieldCalculator(newGpuDevice());
	
	auto computeRequest = session.computeRequest();
	computeRequest.setField(WIRE_FIELD.get());
	computeRequest.setGrid(grid);
	
	Temporary<ComputedField> computed(
		computeRequest.send().wait(ws).getComputedField()
	);
	
	LocalDataRef<Float64Tensor> data = lt->dataService().download(computed.getData()).wait(ws);
	auto fieldOut = readTensor<Tensor<double, 4>>(data.get());
	
	// col major axes 3, nR, nZ, nPhi
	auto zPlane = fieldOut.chip(grid.getNZ() / 2, 2);
	
	TVec3d ref;
	ref(0) = 1;
	ref(1) = 0;
	ref(2) = 0;
	
	for(size_t iPhi = 0; iPhi < grid.getNPhi(); ++iPhi) {
	for(size_t iR = 0; iR < grid.getNR(); ++iR) {
		double r = grid.getRMin() + (grid.getRMax() - grid.getRMin()) / (grid.getNR() - 1) * iR;
		double reference = 2e-7 / r * sin(atan2(1, r));
		TVec3d upscaled = zPlane.chip(iPhi, 2).chip(iR, 1) / reference;
		TensorFixedSize<double, Eigen::Sizes<>> dist = (upscaled - ref).square().sum().sqrt();
		
		KJ_REQUIRE(dist() < 0.01);
	}}
	
}

#endif

}