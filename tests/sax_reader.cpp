#include <goldfish/json_reader.h>
#include <goldfish/sax_reader.h>
#include "unit_test.h"
#include <goldfish/json_reader.h>
#include "dom.h"

namespace goldfish
{
	TEST_CASE(test_conversion_to_double)
	{
		auto r = [](auto input)
		{
			return json::read(stream::read_string_non_owning(input)).as_double();
		};
		test(r("1") == 1);
		test(r("-1") == -1);
		test(r("1.0") == 1);
		test(r("\"1.0\"") == 1);
		test(r("\"1\"") == 1);
		test(r("\"-1\"") == -1);
		expect_exception<bad_variant_access>([&]{ r("[]"); });
	}
	TEST_CASE(test_conversion_to_signed_int)
	{
		auto r = [](auto input)
		{
			return json::read(stream::read_string_non_owning(input)).as_int();
		};
		test(r("1") == 1);
		test(r("-1") == -1);
		test(r("9223372036854775807") == 9223372036854775807ll);

		test(r("\"1\"") == 1);
		test(r("\"-1\"") == -1);

		expect_exception<integer_overflow_while_casting>([&] { r("9223372036854775808"); });
		expect_exception<bad_variant_access>([&] { r("1.0"); });
		expect_exception<bad_variant_access>([&] { r("\"1.0\""); });
		expect_exception<bad_variant_access>([&] { r("[]"); });
	}
	TEST_CASE(test_conversion_to_unsigned_int)
	{
		auto r = [](auto input)
		{
			return json::read(stream::read_string_non_owning(input)).as_uint();
		};
		test(r("1") == 1);
		expect_exception<integer_overflow_while_casting>([&] { r("-1"); });
		expect_exception<bad_variant_access>([&] { r("1.0"); });
		expect_exception<bad_variant_access>([&] { r("[]"); });

		test(r("\"1\"") == 1);
		expect_exception<bad_variant_access>([&] { r("\"1.0\""); });
		expect_exception<integer_overflow_while_casting>([&] { r("\"-1\""); });
	}
	TEST_CASE(test_is_undefined)
	{
		test(json::read(stream::read_string_non_owning("null")).is_undefined());
		test(!json::read(stream::read_string_non_owning("1")).is_undefined());
	}
	TEST_CASE(test_conversion_to_binary)
	{
		auto r = [](auto input)
		{
			return stream::read_all_as_string(json::read(stream::read_string_non_owning(input)).as_binary());
		};
		test(r("\"YW55IGNhcm5hbCBwbGVhc3VyZS4\"") == "any carnal pleasure.");
		test(r("\"SGVsbG8=\"") == "Hello");

		test(stream::read_all_as_string(json::read(stream::read_string_non_owning("\"8000\"")).as_string()) == "8000");
		test(stream::read_all_as_string(json::read(stream::read_string_non_owning("\"8000\"")).as_binary()) == "ÛM4");
		test(json::read(stream::read_string_non_owning("\"8000\"")).as_double() == 8000);
		test(json::read(stream::read_string_non_owning("\"8000\"")).as_int() == 8000);
		test(json::read(stream::read_string_non_owning("\"8000\"")).as_uint() == 8000);
	}
}	