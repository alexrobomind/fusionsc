#include "unit_test.h"
#include <goldfish/match.h>

namespace goldfish
{
	struct A {};
	struct B : A {};
	struct C : A {};

	TEST_CASE(test_match)
	{
		auto first = first_match(
			[](B&&) { return 1; },
			[](A&&) { return 2; },
			[](C&&) { return 3; });

		auto best = best_match(
			[](B&&) { return 1; },
			[](A&&) { return 2; },
			[](C&&) { return 3; });

		test(first(A{}) == 2); test(best(A{}) == 2);
		test(first(B{}) == 1); test(best(B{}) == 1);
		test(first(C{}) == 2); test(best(C{}) == 3);
	}
}

