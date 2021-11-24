#include "w7x.h"
#include "../http.h"

#include <fsc/devices/w7x-test.capnp.h>

#include <catch2/catch_test_macros.hpp>

namespace fsc {
	
namespace devices { namespace w7x {

/**
 * This test cases sets up a dummy coilsDB and then queries the
 * internal proxy for some coils from it. The test cases are
 * defined in w7x-test.capnp
 */
 
TEST_CASE("coilsdb") {
	Library l = newLibrary();
	LibraryThread t = l -> newThread();
	
	auto& ws = t -> waitScope();
	
	auto& network = t -> ioContext().provider -> getNetwork();
	
	auto testData = CDB_TEST.get();
	
	SimpleHttpServer server(network.parseAddress("127.0.0.1"), t, testData.getHttpRoot() /* from w7x-test.capnp */);
	unsigned int port = server.getPort().wait(ws);
	
	CoilsDB::Client cdb = newCoilsDBFromWebservice(network.parseAddress("127.0.0.1", port), t);
	
	for(auto e : testData.getEntries()) {
		auto request = cdb.getCoilRequest();
		request.setId(e.getId());
		
		if(e.isResult()) {
			auto result = request.send().wait(ws);
			auto coil = result.getFilament();
			auto ref  = e.getResult();
			
			REQUIRE(ID::fromReader(coil) == ID::fromReader(ref));
		} else {
			REQUIRE_THROWS(request.send().wait(ws));
		}
	}
}

}}

}