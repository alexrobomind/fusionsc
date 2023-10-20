#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "interpolation.h"

namespace fsc {
	
using Catch::Approx;

TEST_CASE("c1cubic") {
	std::array<double, 4> testCase;
	SECTION("S1") {
		testCase = {1, 0, 0, 0};
	}
	SECTION("S2") {
		testCase = {0, 1, 0, 0};
	}
	SECTION("S3") {
		testCase = {0, 0, 1, 0};
	}
	SECTION("S4") {
		testCase = {0, 0, 0, 1};
	}
	
	C1CubicDeriv<double> interp(testCase[0], testCase[1], testCase[2], testCase[3]);
	REQUIRE(interp(0) == Approx(testCase[0]));
	REQUIRE(interp.d(0) == Approx(testCase[1]));
	REQUIRE(interp(1) == Approx(testCase[2]));
	REQUIRE(interp.d(1) == Approx(testCase[3]));
}

TEST_CASE("c1interp") {
	using Strategy = C1CubicInterpolation<double>;
	using Interpolator = NDInterpolator<1, Strategy>;
	
	Interpolator::Axis ax1(0, 1, 1);
	Interpolator ipol(Strategy(), {ax1});
	
	auto f = [](double x) -> double {
		return x + x*x;
	};
	
	for(int i : kj::range(0, 10)) {
		double dx = 0.3 * i - 4;
		double interp = ipol(f, Vec<double, 1>{dx});
		double fx = f(dx);
		
		REQUIRE(fx == Approx(interp));
	}
}

}