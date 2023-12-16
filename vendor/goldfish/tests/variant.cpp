#include <goldfish/variant.h>
#include "unit_test.h"

namespace goldfish
{
	static_assert(details::index_of<int, int, std::string>::value == 0, "");
	static_assert(std::is_trivially_destructible<variant<int, char>>::value, "");
	static_assert(std::is_trivially_move_constructible<variant<int, char>>::value, "");

	static_assert(details::is_one_of<int, int, std::string>::value, "");

	TEST_CASE(variant_with_single_type)
	{
		variant<int> x;
		test(x.as<int>() == 0);
		x = 3;
		test(x.as<int>() == 3);
	}

	TEST_CASE(copy_variant)
	{
		{
			variant<int, double> x(42);
			auto y = x;
			test(y.as<int>() == 42);

			x = 0.5;
			y = std::move(x);
			test(y.as<double>() == 0.5);
		}

		{
			variant<int, std::string> x("42");
			auto y = x;
			test(y.as<std::string>() == "42");

			x = 42;
			y = x;
			test(y.as<int>() == 42);
		}
	}

	TEST_CASE(variant_with_two_types)
	{
		variant<int, std::string> v;
		v = std::string("foo");
		test(v.is<std::string>());
		test(v.as<std::string>() == "foo");
		v = 3;
		test(v.is<int>());
		test(v.as<int>() == 3);
	}

	TEST_CASE(variant_wrong_type)
	{
		variant<int, std::string> v(3);
		expect_exception<bad_variant_access>([&] { v.as<std::string>(); });
		expect_exception<bad_variant_access>([&] { std::move(v).as<std::string>(); });
		expect_exception<bad_variant_access>([&] { static_cast<const variant<int, std::string>&>(v).as<std::string>(); });
	}
}