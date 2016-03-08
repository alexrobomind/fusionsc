#include <goldfish/json_reader.h>
#include <goldfish/sax_reader.h>
#include "unit_test.h"
#include <goldfish/json_reader.h>
#include <goldfish/dom_reader.h>

namespace goldfish
{
	TEST_CASE(test_filtered_map_empty_map)
	{
		auto json_stream = stream::read_string_literal("{}");

		static const uint64_t schema[] = { 10, 20, 30 };
		auto map = filter_map(json::read(json_stream).as<tags::map>(), schema);

		test(map.read_value_by_index(0) == nullopt);

		skip(map);
	}

	TEST_CASE(test_filtered_map)
	{
		auto json_stream = stream::read_string_literal("{10:1,15:2,\"a\":\"b\",40:3,50:4,60:5,80:6}");

		static const uint64_t schema[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
		auto map = filter_map(json::read(json_stream).as<tags::map>(), schema);

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
		auto json_stream = stream::read_string_literal("{20:1}");

		static const uint64_t schema[] = { 10, 20 };
		auto map = filter_map(json::read(json_stream).as<tags::map>(), schema);

		test(map.read_value_by_index(0) == nullopt);
		skip(map);
	}
}	