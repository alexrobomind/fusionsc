#include <goldfish/debug_checks_writer.h>
#include <goldfish/json_writer.h>
#include "unit_test.h"

namespace goldfish
{
	TEST_CASE(write_multiple_documents_on_same_writer)
	{
		stream::vector_writer output;
		auto writer = json::create_writer(stream::ref(output), debug_check::throw_on_error{});
		writer.write(1ull);
		expect_exception<debug_check::library_missused>([&] { writer.write(1ull); });
	}
	//TEST_CASE(write_on_parent_before_stream_flushed)
	//{
	//	stream::vector_writer output;
	//	auto writer = json::create_writer(stream::ref(output), debug_check::throw_on_error{});
	//	auto writer.write(1ull);
	//	expect_exception<debug_check::library_missused>([&] { writer.write(1ull); });
	//}
}