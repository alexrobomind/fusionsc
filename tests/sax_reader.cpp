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

	TEST_CASE(test_filtered_map_empty_map)
	{
		static const uint64_t schema[] = { 10, 20, 30 };
		auto map = filter_map(json::read(stream::read_string_literal("{}")).as<tags::map>(), schema);

		test(map.read_value_by_index(0) == nullopt);

		skip(map);
	}

	TEST_CASE(test_filtered_map)
	{
		static const uint64_t schema[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
		auto map = filter_map(json::read(
			stream::read_string_literal("{10:1,15:2,\"a\":\"b\",40:3,50:4,60:5,80:6}")).as<tags::map>(), schema);

		// Reading the very first key
		test(dom::load_in_memory(*map.read_value_by_index(0)) == 1ull);

		// Reading index 1 will force to skip the entry 15 and go to entry 40
		test(map.read_value_by_index(1) == nullopt);

		// Reading index 2 will fail because we are already at index 3 of the schema
		test(map.read_value_by_index(2) == nullopt);
		
		// We are currently at index 3 but are asking for index 5, that should skip the pair 40:3 and 50:4 and find 60:5
		test(dom::load_in_memory(*map.read_value_by_index(5)) == 5ull);

		// We ask for index 6, which brings to index 7 (and returns null)
		// Asking for index 7 should return the value on an already read key
		test(map.read_value_by_index(6) == nullopt);
		test(dom::load_in_memory(*map.read_value_by_index(7)) == 6ull);

		// finally, ask for index 8, but we reach the end of the map before we find it
		test(map.read_value_by_index(8) == nullopt);

		skip(map);
	}

	TEST_CASE(filtered_map_skip_while_on_value)
	{
		static const uint64_t schema[] = { 10, 20 };
		auto map = filter_map(json::read(stream::read_string_literal("{20:1}")).as<tags::map>(), schema);

		test(map.read_value_by_index(0) == nullopt);
		skip(map);
	}
}	