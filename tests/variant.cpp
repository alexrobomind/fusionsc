#include "variant.h"
#include "unit_test.h"

namespace gold_fish
{
	static_assert(details::index_of<int, int, std::string>::value == 0, "");
	static_assert(std::is_trivially_copyable<variant<int, char>>::value, "");
	static_assert(std::is_trivially_destructible<variant<int, char>>::value, "");
	static_assert(std::is_trivially_move_constructible<variant<int, char>>::value, "");

	TEST_CASE(variant_with_single_type)
	{
		variant<int> x;
		TEST(x.as<int>() == 0);
		x = 3;
		TEST(x.as<int>() == 3);
	}

	TEST_CASE(variant_with_two_types)
	{
		variant<int, std::string> test;
		test = std::string("foo");
		TEST(test.is<std::string>());
		TEST(test.as<std::string>() == "foo");
		test = 3;
		TEST(test.is<int>());
		TEST(test.as<int>() == 3);
	}
}