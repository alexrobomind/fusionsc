#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "poly.h"

using namespace fsc;

#define RANDPERM (rand() * 0.2 / RAND_MAX - 0.1);

TEST_CASE("triangulate-poly") {
	Tensor<double, 2> poly(5, 2);
	
	poly(0, 0) = 0;
	poly(0, 1) = 0;
	
	poly(1, 0) = 0.5;
	poly(1, 1) = 0.5;
	
	poly(2, 0) = 1;
	poly(2, 1) = 0;
	
	poly(3, 0) = 1;
	poly(3, 1) = 1;
	
	poly(4, 0) = 0;
	poly(4, 1) = 1;
	
	auto result = triangulate(poly);
	// std::cout << result << std::endl;
}