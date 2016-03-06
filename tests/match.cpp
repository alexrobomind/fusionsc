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

		TEST(x(A{}) == 2);
		TEST(x(B{}) == 1);
		TEST(x(C{}) == 2);
	}
}

