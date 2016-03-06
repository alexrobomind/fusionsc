#include "variant.h"
#include "unit_test.h"

namespace gold_fish
{
	static_assert(details::index_of<int, int, std::string>::value == 0, "");
	static_assert(std::is_trivially_copyable<variant<int, char>>::value, "");

	TEST_CASE(variant_one)
	{
		variant<int> x;
		variant<int, std::string> test;
		test = std::string("foo");
		test = 3;
	}
}