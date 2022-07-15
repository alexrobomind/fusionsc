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
	
	SimpleHttpServer server(network.parseAddress("127.0.0.1"), testData.getHttpRoot() /* from w7x-test.capnp */);
	unsigned int port = server.getPort().wait(ws);
	
	CoilsDB::Client cdb = newCoilsDBFromWebservice(kj::str("http://127.0.0.1:", port));
	
	for(auto e : testData.getEntries()) {
		auto request = cdb.getCoilRequest();
		request.setId(e.getId());
		
		if(e.isResult()) {
			auto result = request.send().wait(ws);
			auto ref  = e.getResult();
			
			REQUIRE(ID::fromReader(Filament::Reader(result)) == ID::fromReader(ref));
		} else {
			REQUIRE_THROWS(request.send().wait(ws));
		}
	}
}
 
TEST_CASE("compdb") {
	Library l = newLibrary();
	LibraryThread t = l -> newThread();
	
	auto& ws = t -> waitScope();
	
	auto& network = t -> ioContext().provider -> getNetwork();
	
	auto testData = COMPDB_TEST.get();
	
	SimpleHttpServer server(network.parseAddress("127.0.0.1"), testData.getHttpRoot() /* from w7x-test.capnp */);
	unsigned int port = server.getPort().wait(ws);
	
	ComponentsDB::Client cdb = newComponentsDBFromWebservice(kj::str("http://127.0.0.1:", port));
	
	for(auto e : testData.getEntries()) {
		auto request = cdb.getMeshRequest();
		request.setId(e.getId());
		
		if(e.isResult()) {
			auto result = request.send().wait(ws);
			Mesh::Reader mesh = result;
			Mesh::Reader ref  = e.getResult();
			
			KJ_REQUIRE(ID::fromReader(mesh) == ID::fromReader(ref), mesh, ref);
		} else {
			REQUIRE_THROWS(request.send().wait(ws));
		}
	}
}
 
TEST_CASE("preheat") {
	Library l = newLibrary();
	LibraryThread t = l -> newThread();
	
	auto& ws = t -> waitScope();
	
	// Get fields to preheat
	auto fields = preheatFields(Temporary<W7XCoilSet>().asReader());
}

}}

}