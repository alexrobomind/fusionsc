#include <goldfish/debug_checks_writer.h>
#include <goldfish/json_writer.h>
#include "unit_test.h"

namespace goldfish
{
	struct library_misused {};
	struct throw_on_error
	{
		static void on_error() { throw library_misused{}; }
	};

	TEST_CASE(write_multiple_documents_on_same_writer)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{});
		writer.write(1ull);
		expect_exception<library_misused>([&] { writer.write(1ull); });
	}
	TEST_CASE(write_on_parent_before_stream_flushed)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto stream = array.start_string();
		expect_exception<library_misused>([&] { array.append(); });
	}
	TEST_CASE(write_to_stream_after_flush)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto stream = array.start_string();
		stream.flush();
		expect_exception<library_misused>([&] { stream::write(stream, 'a'); });
	}
	TEST_CASE(flush_stream_twice)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto stream = array.start_string();
		stream.flush();
		expect_exception<library_misused>([&] { stream.flush(); });
	}
	TEST_CASE(flush_stream_without_writing_all)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto stream = array.start_string(2);
		stream::write(stream, 'a');
		expect_exception<library_misused>([&] { stream.flush(); });
	}
	TEST_CASE(write_too_much_to_stream)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto stream = array.start_string(1);
		stream::write(stream, 'a');
		expect_exception<library_misused>([&] { stream::write(stream, 'b'); });
	}

	TEST_CASE(write_on_parent_before_array_flushed)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto array = writer.start_array();
		expect_exception<library_misused>([&] { writer.append(); });
	}
	TEST_CASE(write_to_array_after_flush)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto array = writer.start_array();
		array.flush();
		expect_exception<library_misused>([&] { array.append(); });
	}
	TEST_CASE(append_to_array_without_writing)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto array = writer.start_array();
		array.append();
		expect_exception<library_misused>([&] { array.flush(); });
	}
	TEST_CASE(flush_array_twice)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto array = writer.start_array();
		array.flush();
		expect_exception<library_misused>([&] { array.flush(); });
	}
	TEST_CASE(flush_array_without_writing_all)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto array = writer.start_array(2);
		array.write(1ull);
		expect_exception<library_misused>([&] { array.flush(); });
	}
	TEST_CASE(write_too_much_to_array)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto array = writer.start_array(1);
		array.write(1ull);
		expect_exception<library_misused>([&] { array.append(); });
	}

	TEST_CASE(write_on_parent_before_map_flushed)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map();
		expect_exception<library_misused>([&] { writer.append(); });
	}
	TEST_CASE(write_to_map_after_flush)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map();
		map.flush();
		expect_exception<library_misused>([&] { map.append_key(); });
	}
	TEST_CASE(append_to_map_without_writing)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map();
		map.append_key();
		expect_exception<library_misused>([&] { map.append_value(); });
	}
	TEST_CASE(flush_map_twice)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map();
		map.flush();
		expect_exception<library_misused>([&] { map.flush(); });
	}
	TEST_CASE(flush_map_without_writing_all)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map(2);
		map.write_key(1ull);
		map.write_value(1ull);
		expect_exception<library_misused>([&] { map.flush(); });
	}
	TEST_CASE(write_too_much_to_map)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map(1);
		map.write_key(1ull);
		map.write_value(1ull);
		expect_exception<library_misused>([&] { map.append_key(); });
	}
	TEST_CASE(write_value_to_map_when_key_expected)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map();
		expect_exception<library_misused>([&] { map.append_value(); });
	}
	TEST_CASE(write_key_to_map_when_value_expected)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map();
		map.write_key(1ull);
		expect_exception<library_misused>([&] { map.append_key(); });
	}
	TEST_CASE(flush_map_when_value_expected)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), throw_on_error{}).start_array();
		auto map = writer.start_map();
		map.write_key(1ull);
		expect_exception<library_misused>([&] { map.flush(); });
	}
}