#include <goldfish/debug_checks_writer.h>
#include <goldfish/json_writer.h>
#include "unit_test.h"

namespace goldfish
{
	TEST_CASE(write_multiple_documents_on_same_writer)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{});
		writer.write(1ull);
		expect_exception<debug_checks::library_missused>([&] { writer.write(1ull); });
	}
	TEST_CASE(write_on_parent_before_stream_flushed)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto stream = array.append().write(tags::string{});
		expect_exception<debug_checks::library_missused>([&] { array.append(); });
	}
	TEST_CASE(write_to_stream_after_flush)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto stream = array.append().write(tags::string{});
		stream.flush();
		expect_exception<debug_checks::library_missused>([&] { stream::write(stream, 'a'); });
	}
	TEST_CASE(flush_stream_twice)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto stream = array.append().write(tags::string{});
		stream.flush();
		expect_exception<debug_checks::library_missused>([&] { stream.flush(); });
	}
	TEST_CASE(flush_stream_without_writing_all)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto stream = array.append().write(tags::string{}, 2);
		stream::write(stream, 'a');
		expect_exception<debug_checks::library_missused>([&] { stream.flush(); });
	}
	TEST_CASE(write_too_much_to_stream)
	{
		stream::vector_writer output;
		auto array = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto stream = array.append().write(tags::string{}, 1);
		stream::write(stream, 'a');
		expect_exception<debug_checks::library_missused>([&] { stream::write(stream, 'b'); });
	}

	TEST_CASE(write_on_parent_before_array_flushed)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto array = writer.append().write(tags::array{});
		expect_exception<debug_checks::library_missused>([&] { writer.append(); });
	}
	TEST_CASE(write_to_array_after_flush)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto array = writer.append().write(tags::array{});
		array.flush();
		expect_exception<debug_checks::library_missused>([&] { array.append(); });
	}
	TEST_CASE(append_to_array_without_writing)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto array = writer.append().write(tags::array{});
		array.append();
		expect_exception<debug_checks::library_missused>([&] { array.flush(); });
	}
	TEST_CASE(flush_array_twice)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto array = writer.append().write(tags::array{});
		array.flush();
		expect_exception<debug_checks::library_missused>([&] { array.flush(); });
	}
	TEST_CASE(flush_array_without_writing_all)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto array = writer.append().write(tags::array{}, 2);
		array.append().write(1ull);
		expect_exception<debug_checks::library_missused>([&] { array.flush(); });
	}
	TEST_CASE(write_too_much_to_array)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto array = writer.append().write(tags::array{}, 1);
		array.append().write(1ull);
		expect_exception<debug_checks::library_missused>([&] { array.append(); });
	}

	TEST_CASE(write_on_parent_before_map_flushed)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{});
		expect_exception<debug_checks::library_missused>([&] { writer.append(); });
	}
	TEST_CASE(write_to_map_after_flush)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{});
		map.flush();
		expect_exception<debug_checks::library_missused>([&] { map.append_key(); });
	}
	TEST_CASE(append_to_map_without_writing)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{});
		map.append_key();
		expect_exception<debug_checks::library_missused>([&] { map.append_value(); });
	}
	TEST_CASE(flush_map_twice)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{});
		map.flush();
		expect_exception<debug_checks::library_missused>([&] { map.flush(); });
	}
	TEST_CASE(flush_map_without_writing_all)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{}, 2);
		map.append_key().write(1ull);
		map.append_value().write(1ull);
		expect_exception<debug_checks::library_missused>([&] { map.flush(); });
	}
	TEST_CASE(write_too_much_to_map)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{}, 1);
		map.append_key().write(1ull);
		map.append_value().write(1ull);
		expect_exception<debug_checks::library_missused>([&] { map.append_key(); });
	}
	TEST_CASE(write_value_to_map_when_key_expected)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{});
		expect_exception<debug_checks::library_missused>([&] { map.append_value(); });
	}
	TEST_CASE(write_key_to_map_when_value_expected)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{});
		map.append_key().write(1ull);
		expect_exception<debug_checks::library_missused>([&] { map.append_key(); });
	}
	TEST_CASE(flush_map_when_value_expected)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_checks::throw_on_error{}).write(tags::array{});
		auto map = writer.append().write(tags::map{});
		map.append_key().write(1ull);
		expect_exception<debug_checks::library_missused>([&] { map.flush(); });
	}
}