#include <catch2/catch_test_macros.hpp>

#include <fsc/magnetics-test.capnp.h>

#include "magnetics.h"
#include "local.h"
#include "data.h"

namespace fsc {

TEST_CASE("build-field") {
	auto field = WIRE_FIELD.get();
	
	Library l = newLibrary();
	LibraryThread lt = l -> newThread();
	auto& ws = lt->waitScope();
	
	Temporary<ToroidalGrid> grid;
	
	auto calc = newFieldCalculator(lt);
	
	auto sessRequest = calc.getRequest();
	sessRequest.setGrid(TEST_GRID.get());
	
	auto session = sessRequest.send().wait(ws).getSession();
	
	auto computeRequest = session.computeRequest();
	computeRequest.setField(WIRE_FIELD.get());
	
	Temporary<ComputedField> computed(
		computeRequest.send().wait(ws).getComputedField()
	);
}

}