#include <catch2/catch_test_macros.hpp>

#include <fsc/geometry-test.capnp.h>

#include "magnetics.h"
#include "local.h"
#include "data.h"
#include "tensor.h"
#include "geometry.h"

namespace fsc {

namespace {
	auto newCpuDevice() {
		return CPUDevice::create(CPUDevice::estimateNumThreads());
	}
}

TEST_CASE("intersect") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt -> waitScope();
	
	DataRef<Mesh>::Client cube = lt->dataService().publish(TEST_CUBE.get());
	
	GeometryLib::Client geoLib = newGeometryLib(newCpuDevice());
	
	auto indexRequest = geoLib.indexRequest();
	indexRequest.initGeometry().setMesh(cube);
	
	auto grid = indexRequest.initGrid();
	grid.setNX(1);
	grid.setXMin(-10);
	grid.setXMax(10);
	
	grid.setNY(1);
	grid.setYMin(-10);
	grid.setYMax(10);
	
	grid.setNZ(1);
	grid.setZMin(-10);
	grid.setZMax(10);
	
	auto resp1 = indexRequest.send().wait(ws);
	auto indexed = resp1.getIndexed();
	KJ_DBG("Index OK");
	
	auto intersectRequest = geoLib.intersectRequest();
	intersectRequest.setGeometry(indexed);
	auto p1 = intersectRequest.initPStart();
	auto p2 = intersectRequest.initPEnd();
	
	p1.setShape({3});
	p1.setData({0, 0, 0});
	p2.setShape({3});
	p2.setData({2, 0, 0});
	
	auto response = intersectRequest.send().wait(ws);
	KJ_DBG(response);
}

TEST_CASE("transform-cube") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	
	DataRef<Mesh>::Client cube = lt->dataService().publish(TEST_CUBE.get());
	
	Temporary<Geometry> geometry;
	
	auto tags = geometry.initTags(1);
	tags[0].setName("tag");
	tags[0].initValue().setUInt64(25);
	
	auto turned = geometry.initTransformed().initTurned();
	turned.setAxis({0, 0, 1});
	turned.setCenter({0, 0, 0});
	turned.getAngle().setRad(3.14159265358979323846);
	
	turned.initNode().initLeaf().setMesh(cube);
	
	GeometryLib::Client geoLib = newGeometryLib(newCpuDevice());
	
	auto mergeRequest = geoLib.mergeRequest();
	mergeRequest.setRef(lt->dataService().publish(geometry.asReader()));
	
	auto mergeResult = mergeRequest.send();
	auto transformedCube = lt->dataService().download(mergeResult.getRef()).wait(lt->waitScope());
}

TEST_CASE("quad") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt -> waitScope();
	GeometryLib::Client geoLib = newGeometryLib(newCpuDevice());
		
	Temporary<Geometry> geometry;
	auto qm = geometry.initQuadMesh();
	
	{
		Temporary<Float64Tensor> t;
		t.setData({0,0,0,0,0,1});
		t.setShape({2, 1, 3});
		qm.setVertices(getActiveThread().dataService().publish(t.asReader()));
	}
	
	qm.setWrapU(false);
	qm.setWrapV(true);
	
	auto req1 = geoLib.mergeRequest();
	req1.setNested(geometry);
	auto resp = req1.send().wait(ws);
	
	KJ_DBG(getActiveThread().dataService().download(resp.getRef()).wait(ws).get());
}

TEST_CASE("index-cube") {
	// Note: This test contains a lot of manual extraction of values into unpacked numbers (x, y, z, etc.)
	// This is to ensure a different coding style than the geometry utilities and avoid using them as much
	// as possible. This should hoepfully minimize the likelihood of logic errors in the geometry code
	// making it also into the test.
	
	auto l  = newLibrary();
	auto lt = l -> newThread();
	auto& ws = lt -> waitScope();
	GeometryLib::Client geoLib = newGeometryLib(newCpuDevice());
	
	Mesh::Reader geo = TEST_CUBE.get();
	
	// Check some assumptions about geometry
	KJ_REQUIRE(geo.isPolyMesh());
	auto polyIndices = geo.getPolyMesh();
	for(uint32_t iPoly : kj::range(0, polyIndices.size())) {
		KJ_REQUIRE(polyIndices[iPoly] == 4 * iPoly, iPoly, "Assuming only rectangular faces");
	}
	uint32_t nPolys = polyIndices.size() - 1;
	
	auto mergeRequest = geoLib.mergeRequest();
	mergeRequest.setMesh(lt->dataService().publish(geo));
	auto mergeResult = mergeRequest.send();
	
	Temporary<CartesianGrid> grid;
	grid.setXMin(-1);
	grid.setYMin(-1);
	grid.setZMin(-1);
	
	grid.setXMax(1);
	grid.setYMax(1);
	grid.setZMax(1);
	
	grid.setNX(4);
	grid.setNY(4);
	grid.setNZ(4);
	
	auto indexRequest = geoLib.indexRequest();
	indexRequest.initGeometry().setMerged(mergeResult.getRef());
	indexRequest.setGrid(grid);
	
	auto indexResult = indexRequest.send().wait(ws);
	auto indexDataRef = lt->dataService().download(indexResult.getIndexed().getData()).wait(ws);
	auto indexData = indexDataRef.get();
	
	double sizeX = (grid.getXMax() - grid.getXMin()) / grid.getNX();
	double sizeY = (grid.getYMax() - grid.getYMin()) / grid.getNY();
	double sizeZ = (grid.getZMax() - grid.getZMin()) / grid.getNZ();
	
	for(uint32_t iX = 0; iX < grid.getNX(); ++iX) {
	for(uint32_t iY = 0; iY < grid.getNY(); ++iY) {
	for(uint32_t iZ = 0; iZ < grid.getNZ(); ++iZ) {
	
		uint32_t linearIndex = (iX * grid.getNY() + iY) * grid.getNZ() + iZ;
		
		double cellX = grid.getXMin() + iX * sizeX;
		double cellY = grid.getYMin() + iY * sizeY;
		double cellZ = grid.getZMin() + iZ * sizeZ;
		
		for(uint32_t iPoly = 0; iPoly < nPolys; ++iPoly) {
			// Check whether the polygon should be in this cell
			const double inf = std::numeric_limits<double>::infinity();
			double xMin = inf; double xMax = -inf;
			double yMin = inf; double yMax = -inf;
			double zMin = inf; double zMax = -inf;
			
			for(uint32_t iPoint = 4 * iPoly; iPoint < 4 * (iPoly + 1); ++iPoint) {
				uint32_t pointIdx = geo.getIndices()[iPoint];
				double x = geo.getVertices().getData()[3 * pointIdx + 0];
				double y = geo.getVertices().getData()[3 * pointIdx + 1];
				double z = geo.getVertices().getData()[3 * pointIdx + 2];
				
				xMin = std::min(x, xMin);
				yMin = std::min(y, yMin);
				zMin = std::min(z, zMin);
				
				xMax = std::max(x, xMax);
				yMax = std::max(y, yMax);
				zMax = std::max(z, zMax);
			}
			
			bool shouldBeInCell = true;
			
			if(cellX > xMax || cellX + sizeX < xMin)
				shouldBeInCell = false;
			
			if(cellY > yMax || cellY + sizeY < yMin)
				shouldBeInCell = false;
			
			if(cellZ > zMax || cellZ + sizeZ < zMin)
				shouldBeInCell = false;
			
			// Check whether the polygon is in this cell
			bool isInCell = false;
			for(auto entry : indexData.getGridContents().getData()[linearIndex]) {
				KJ_REQUIRE(entry.getMeshIndex() == 0);
				if(entry.getElementIndex() == iPoly)
					isInCell = true;
			}
			
			KJ_REQUIRE(shouldBeInCell == isInCell, iPoly);
		}
		
	}}}
}


}