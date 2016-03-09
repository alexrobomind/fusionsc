#include <goldfish/json_reader.h>
#include <goldfish/sax_reader.h>
#include "unit_test.h"
#include <goldfish/json_reader.h>
#include <goldfish/dom_reader.h>

namespace goldfish
{
	TEST_CASE(test_conversion_to_double)
	{
		auto r = [](auto input)
		{
			return json::read(stream::read_string_literal(input)).as<tags::floating_point>();
		};
		test(r("1") == 1);
		test(r("-1") == -1);
		test(r("1.0") == 1);
		expect_exception<bad_variant_access>([&]{ r("[]"); });
	}
	TEST_CASE(test_conversion_to_signed_int)
	{
		auto r = [](auto input)
		{
			return json::read(stream::read_string_literal(input)).as<tags::signed_int>();
		};
		test(r("1") == 1);
		test(r("-1") == -1);
		test(r("9223372036854775807") == 9223372036854775807ll);
		expect_exception<integer_overflow>([&] { r("9223372036854775808"); });
		expect_exception<bad_variant_access>([&] { r("1.0"); });
		expect_exception<bad_variant_access>([&] { r("[]"); });
	}
	TEST_CASE(test_conversion_to_unsigned_int)
	{
		auto r = [](auto input)
		{
			return json::read(stream::read_string_literal(input)).as<tags::unsigned_int>();
		};
		test(r("1") == 1);
		expect_exception<bad_variant_access>([&] { r("-1"); });
		expect_exception<bad_variant_access>([&] { r("1.0"); });
		expect_exception<bad_variant_access>([&] { r("[]"); });
	}
	TEST_CASE(test_conversion_to_binary)
	{
		auto r = [](auto input)
		{
			return stream::read_all_as_string(json::read(stream::read_string_literal(input)).as<tags::byte_string>());
		};
		test(r("\"YW55IGNhcm5hbCBwbGVhc3VyZS4\"") == "any carnal pleasure.");
	}
}	