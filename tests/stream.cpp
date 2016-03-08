#include <goldfish/stream.h>
#include "unit_test.h"

namespace goldfish { namespace stream {

static_assert(is_reader<empty>::value, "empty is a reader");
static_assert(!is_reader<vector_writer>::value, "vector_writer is not a reader");
static_assert(!is_writer<empty>::value, "empty is not a writer");
static_assert(is_writer<vector_writer>::value, "vector_writer is a writer");

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