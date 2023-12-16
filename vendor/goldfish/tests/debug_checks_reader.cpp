#include <goldfish/debug_checks_reader.h>
#include <goldfish/json_reader.h>
#include "unit_test.h"

namespace goldfish
{
	struct library_misused {};
	struct throw_on_error
	{
		static void on_error() { throw library_misused{}; }
	};

	TEST_CASE(reading_parent_before_stream_end)
	{
		auto document = json::read(stream::read_string("[\"hello\"]"), throw_on_error{}).as_array();

		auto string = document.read()->as_string();
		test(stream::read<char>(string) == 'h');
		test(stream::seek(string, 1) == 1);
		expect_exception<library_misused>([&] { document.read(); });
	}
	TEST_CASE(reading_parent_after_reading_all_ok)
	{
		auto document = json::read(stream::read_string("[\"hello\"]"), throw_on_error{}).as_array();

		auto string = document.read()->as_string();
		test(stream::read_all_as_string(string) == "hello");
		test(document.read() == nullopt);
	}
	TEST_CASE(reading_parent_after_seeking_to_exactly_end_throws)
	{
		auto document = json::read(stream::read_string("[\"hello\"]"), throw_on_error{}).as_array();

		auto string = document.read()->as_string();
		test(stream::seek(string, 5) == 5);
		expect_exception<library_misused>([&] { document.read(); });
	}
	TEST_CASE(reading_parent_after_seeking_past_end_ok)
	{
		auto document = json::read(stream::read_string("[\"hello\"]"), throw_on_error{}).as_array();

		auto string = document.read()->as_string();
		test(stream::seek(string, 6) == 5);
		test(document.read() == nullopt);
	}

	TEST_CASE(reading_parent_before_end_of_array_throws)
	{
		auto document = json::read(stream::read_string("[[1, 2]]"), throw_on_error{}).as_array();

		auto array = document.read()->as_array();
		test(array.read()->as_uint64() == 1);
		expect_exception<library_misused>([&] { document.read(); });
	}
	TEST_CASE(reading_parent_at_exactly_end_of_array_throws)
	{
		auto document = json::read(stream::read_string("[[1, 2]]"), throw_on_error{}).as_array();

		auto array = document.read()->as_array();
		test(array.read()->as_uint64() == 1);
		test(array.read()->as_uint64() == 2);
		expect_exception<library_misused>([&] { document.read(); });
	}
	TEST_CASE(reading_parent_passed_end_of_array_ok)
	{
		auto document = json::read(stream::read_string("[[1, 2]]"), throw_on_error{}).as_array();

		auto array = document.read()->as_array();
		test(array.read()->as_uint64() == 1);
		test(array.read()->as_uint64() == 2);
		test(array.read() == nullopt);
		test(document.read() == nullopt);
	}

	TEST_CASE(reading_parent_before_end_of_map_throws)
	{
		auto document = json::read(stream::read_string("[{\"a\":1, \"b\":2}]"), throw_on_error{}).as_array();

		auto map = document.read()->as_map();
		test(stream::read_all_as_string(map.read_key()->as_string()) == "a");
		expect_exception<library_misused>([&] { document.read(); });
	}
	TEST_CASE(reading_parent_at_exactly_end_of_map_throws)
	{
		auto document = json::read(stream::read_string("[{\"a\":1, \"b\":2}]"), throw_on_error{}).as_array();

		auto map = document.read()->as_map();
		test(stream::read_all_as_string(map.read_key()->as_string()) == "a");
		test(map.read_value().as_uint64() == 1);
		test(stream::read_all_as_string(map.read_key()->as_string()) == "b");
		test(map.read_value().as_uint64() == 2);
		expect_exception<library_misused>([&] { document.read(); });
	}
	TEST_CASE(reading_parent_passed_end_of_map_ok)
	{
		auto document = json::read(stream::read_string("[{\"a\":1, \"b\":2}]"), throw_on_error{}).as_array();

		auto map = document.read()->as_map();
		test(stream::read_all_as_string(map.read_key()->as_string()) == "a");
		test(map.read_value().as_uint64() == 1);
		test(stream::read_all_as_string(map.read_key()->as_string()) == "b");
		test(map.read_value().as_uint64() == 2);
		test(map.read_key() == nullopt);

		test(document.read() == nullopt);
	}
	TEST_CASE(reading_value_before_finishing_key_in_map)
	{
		auto document = json::read(stream::read_string("[{\"a\":1, \"b\":2}]"), throw_on_error{}).as_array();

		auto map = document.read()->as_map();
		map.read_key();
		expect_exception<library_misused>([&] { map.read_value(); });
	}
	TEST_CASE(reading_key_before_finishing_value_in_map)
	{
		auto document = json::read(stream::read_string("[{\"a\":\"1\", \"b\":2}]"), throw_on_error{}).as_array();

		auto map = document.read()->as_map();
		test(stream::read_all_as_string(map.read_key()->as_string()) == "a");
		map.read_value();
		expect_exception<library_misused>([&] { map.read_key(); });
	}
	TEST_CASE(reading_value_instead_of_key_in_map)
	{
		auto document = json::read(stream::read_string("[{\"a\":1, \"b\":2}]"), throw_on_error{}).as_array();

		auto map = document.read()->as_map();
		expect_exception<library_misused>([&] { map.read_value(); });
	}
	TEST_CASE(reading_key_instead_of_value_in_map)
	{
		auto document = json::read(stream::read_string("[{\"a\":1, \"b\":2}]"), throw_on_error{}).as_array();

		auto map = document.read()->as_map();
		test(stream::read_all_as_string(map.read_key()->as_string()) == "a");
		expect_exception<library_misused>([&] { map.read_key(); });
	}
}