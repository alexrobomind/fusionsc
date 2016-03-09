#include <goldfish/buffered_stream.h>

#include "unit_test.h"

namespace goldfish { namespace stream
{
	TEST_CASE(test_buffered)
	{
		vector_writer x;
		auto stream = buffer<2>(ref(x));
		stream.write<uint8_t>(1);
		test(x.data.empty());

		stream.write<uint8_t>(2);
		test(x.data.empty());

		stream.flush();
		test(x.data == std::vector<uint8_t>{1, 2});
	}
}}