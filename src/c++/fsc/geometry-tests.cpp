#include <catch2/catch_test_macros.hpp>

#include <fsc/geometry-test.capnp.h>

#include "magnetics.h"
#include "local.h"
#include "data.h"
#include "tensor.h"
#include "geometry.h"

namespace fsc {

TEST_CASE("transform-cube") {
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	
	KJ_LOG(WARNING, "Publishing");
	DataRef<Mesh>::Client cube = lt->dataService().publish(TEST_CUBE.get());
	KJ_LOG(WARNING, "Published");
	
	Temporary<Geometry> geometry;
	
	auto tags = geometry.initTags(1);
	tags[0].setName("tag");
	tags[0].initValue().setUInt64(25);
	
	auto turned = geometry.initTransformed().initTurned();
	turned.setAxis({0, 0, 1});
	turned.setCenter({0, 0, 0});
	turned.setAngle(3.14159265358979323846);
	
	turned.initNode().initLeaf().setMesh(cube);
	
	GeometryLib::Client geoLib = createGeometryLib(lt);
	
	KJ_LOG(WARNING, "Preparing merge request");
	auto mergeRequest = geoLib.mergeRequest();
	mergeRequest.setRef(lt->dataService().publish(geometry.asReader()));
	
	KJ_LOG(WARNING, "Sending merge request");
	auto mergeResult = mergeRequest.send();
	KJ_LOG(WARNING, "Downloading cube");
	auto transformedCube = lt->dataService().download(mergeResult.getRef()).wait(lt->waitScope());
	
	KJ_LOG(WARNING, transformedCube.get());
}

}