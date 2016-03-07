#include "unit_test.h"
#include <gold_fish/match.h>

namespace gold_fish
{
	struct A {};
	struct B : A {};
	struct C : A {};

	TEST_CASE(test_match)
	{
		auto x = first_match(
			[](B&&) { return 1; },
			[](A&&) { return 2; }
		);

		test(x(A{}) == 2);
		test(x(B{}) == 1);
		test(x(C{}) == 2);
	}
}

