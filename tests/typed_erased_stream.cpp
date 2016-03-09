#include <goldfish/typed_erased_stream.h>
#include "unit_test.h"

namespace goldfish { namespace stream
{
	TEST_CASE(test_reader_erasure)
	{
		auto s = erase_type(read_string_literal("abcdef"));
		test(read<char>(s) == 'a');
		test(stream::skip(s, 2) == 2);
		test(read<char>(s) == 'd');

		auto t = std::move(s);
		test(stream::skip(t, 3) == 2);
	}
}}