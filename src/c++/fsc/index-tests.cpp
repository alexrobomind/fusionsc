#include <catch2/catch_test_macros.hpp>
#include <fsc/http.capnp.h>

#include <kj/debug.h>

#include "index.h"

namespace fsc {

TEST_CASE("indexBuild") {
	Library l = newLibrary();
	LibraryThread th = l -> newThread();
	
	kj::WaitScope& ws = th -> waitScope();
	
	KDTreeService::Client ts = newKDTreeService();
	
	auto req = ts.buildSimpleRequest();
	req.getPoints().setShape({3, 3});
	req.getPoints().setData({0, 0, 0, 0, 0, 1, 0, 1, 0});
	
	auto treeRef = req.send().getRef();
	auto dl = th -> dataService().download(treeRef).wait(ws);
	
	KJ_DBG(dl.get());
}

}
