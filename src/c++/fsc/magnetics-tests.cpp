#include <catch2/catch_test_macros.hpp>


#include <fsc/magnetics-test.capnp.h>
#include <fsc/magnetics-test.capnp.cu.h>

#include "magnetics.h"
#include "local.h"
#include "data.h"
#include "tensor.h"

namespace fsc {
	
TEST_CASE("build-field-cancel") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
		
	auto grid = TEST_GRID.get();
	FieldCalculator::Client session = newFieldCalculator(CPUDevice::create(CPUDevice::estimateNumThreads()));
	
	auto computeRequest = session.computeRequest();
	computeRequest.setField(WIRE_FIELD.get());
	computeRequest.setGrid(grid);
	
	{
		auto comp = computeRequest.sendForPipeline().getComputedField();
		
		ws.poll(10);
	}
}

TEST_CASE("build-field") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
		
	auto grid = TEST_GRID.get();
	FieldCalculator::Client session = newFieldCalculator(CPUDevice::create(CPUDevice::estimateNumThreads()));
	
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
		
		// KJ_DBG(dist());
		KJ_REQUIRE(dist() < 0.01);
	}}
	
	// Check field evaluation
	
	auto evalRequest = session.evaluateXyzRequest();
	evalRequest.getField().setComputedField(computed);
	//evalRequest.setField(WIRE_FIELD.get());
	
	// KJ_DBG(data.get());
	
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
			
			double bR = bx * std::cos(phi) + by * std::sin(phi);
			double bTor = by * std::cos(phi) - bx * std::sin(phi);
			
			// KJ_DBG(bx, refX, by, refY, bz);
			// KJ_DBG(r, phi, bR, bTor, ref);
			
			KJ_REQUIRE(std::abs(bx / ref - refX / ref) < 0.05);
			KJ_REQUIRE(std::abs(by / ref - refY / ref) < 0.05);
			KJ_REQUIRE(std::abs(bz / ref) < 0.05);
		}
	}			
}

TEST_CASE("build-field-interp") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	auto grid1 = TEST_GRID.get();
	auto grid2 = TEST_GRID2.get();
	FieldCalculator::Client session = newFieldCalculator(CPUDevice::create(CPUDevice::estimateNumThreads()));
	
	// Compute field on standard grid
	auto cr1 = session.computeRequest();
	
	auto fieldIn = cr1.initField();
	
	SECTION("base") {}
	
	SECTION("shift") {
		auto t = fieldIn.initTransformed();
		auto s = t.initShifted();
		s.setShift({0, 0, 1});
		fieldIn = s.getNode().initLeaf();
	}
	
	SECTION("turn") {
		auto t = fieldIn.initTransformed();
		auto tu = t.initTurned();
		tu.setAxis({0, 0, 1});
		tu.getAngle().setDeg(30);
		
		fieldIn = tu.getNode().initLeaf();
	}
	
	fieldIn.setNested(WIRE_FIELD.get());
	cr1.setGrid(grid1);
	
	KJ_DBG(cr1.getField());
	
	auto result1 = cr1.send().wait(ws);
	
	auto cr2 = session.computeRequest();
	cr2.setField(WIRE_FIELD.get());
	cr2.setGrid(grid2);
	
	auto result2 = cr2.send().wait(ws);
	
	auto cr3 = session.computeRequest();
	auto sum = cr3.getField().initSum(2);
	sum[0].setComputedField(result1.getComputedField());
	sum[1].setComputedField(result2.getComputedField());
	cr3.setGrid(grid1);
	
	auto result3 = cr3.send().wait(ws);
	
	LocalDataRef<Float64Tensor> ref1 = lt->dataService().download(result1.getComputedField().getData()).wait(ws);
	LocalDataRef<Float64Tensor> ref2 = lt->dataService().download(result3.getComputedField().getData()).wait(ws);
	
	auto data1 = ref1.get().getData();
	auto data2 = ref2.get().getData();
		
	KJ_REQUIRE(data1.size() == data2.size());
	for(auto i : kj::indices(data1)) {
		KJ_REQUIRE(std::abs(2 * data1[i] - data2[i]) < 0.05);
	}
}

#ifdef FSC_WITH_CUDA

TEST_CASE("build-field-gpu") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	auto grid = TEST_GRID.get();
	FieldCalculator::Client session = newFieldCalculator(newGpuDevice());
	
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
