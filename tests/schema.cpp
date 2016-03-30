#include <goldfish/schema.h>
#include <goldfish/json_reader.h>
#include "dom.h"
#include "unit_test.h"

namespace goldfish
{
	struct library_misused {};
	struct throw_on_error
	{
		static void on_error() { throw library_misused{}; }
	};

	TEST_CASE(test_filtered_map_empty_map)
	{
		auto map = json::read(stream::read_string("{}")).as_map("10", "20", "30");

		test(map.read_by_schema_index(0) == nullopt);

		seek_to_end(map);
	}

	TEST_CASE(test_filtered_map)
	{
		auto map = json::read(
			stream::read_string("{\"10\":1,\"15\":2,\"a\":\"b\",\"40\":3,\"50\":4,\"60\":5,\"80\":6}")).
			as_map("10", "20", "30", "40", "50", "60", "70", "80", "90");

		// Reading the very first key
		test(dom::load_in_memory(*map.read_by_schema_index(0)) == 1ull);

		// Reading index 1 will force to skip the entry 15 and go to entry 40
		test(map.read_by_schema_index(1) == nullopt);

		// Reading index 2 will fail because we are already at index 3 of the schema
		test(map.read_by_schema_index(2) == nullopt);

		// We are currently at index 3 but are asking for index 5, that should skip the pair 40:3 and 50:4 and find 60:5
		test(dom::load_in_memory(*map.read_by_schema_index(5)) == 5ull);

		// We ask for index 6, which brings to index 7 (and returns null)
		// Asking for index 7 should return the value on an already read key
		test(map.read_by_schema_index(6) == nullopt);
		test(dom::load_in_memory(*map.read_by_schema_index(7)) == 6ull);

		// finally, ask for index 8, but we reach the end of the map before we find it
		test(map.read_by_schema_index(8) == nullopt);

		seek_to_end(map);
	}

	TEST_CASE(filtered_map_skip_while_on_value)
	{
		auto map = json::read(stream::read_string("{\"20\":1}")).as_map("10", "20");

		test(map.read_by_schema_index(0) == nullopt);
		seek_to_end(map);
	}

	TEST_CASE(test_filtered_map_by_value)
	{
		auto map = json::read(stream::read_string("{\"B\":1}")).as_map("A", "B");
		test(dom::load_in_memory(*map.read("B")) == 1ull);
		seek_to_end(map);
	}

	TEST_CASE(test_missing_seek_to_end_err)
	{
		auto a = json::read(stream::read_string("[{}]"), throw_on_error{}).as_array();
		
		auto map = a.read().value().as_map("A", "B");
		test(map.read("A") == nullopt);

		// Even though in this particular example, the map reached the end, it's still invalid to read from a
		// because map.read("A") might have returned null because "A" wasn't found (and "B" might still be in the map)
		expect_exception<library_misused>([&] { a.read(); });
	}
}