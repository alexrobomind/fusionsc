#include <goldfish/stream.h>
#include "unit_test.h"

namespace goldfish { namespace stream {

static_assert(is_reader<empty>::value, "empty is a reader");
static_assert(!is_reader<vector_writer>::value, "vector_writer is not a reader");
static_assert(!is_writer<empty>::value, "empty is not a writer");
static_assert(is_writer<vector_writer>::value, "vector_writer is a writer");

TEST_CASE(test_skip)
{
	struct fake_stream
	{
		fake_stream(size_t size)
			: m_size(size)
		{}

		size_t m_size;
		size_t m_calls = 0;

		size_t read_buffer(buffer_ref buffer)
		{
			// pretend we filled buffer with data
			++m_calls;
			auto cb = std::min(buffer.size(), m_size);
			m_size -= cb;
			return cb;
		}
	};
	static const size_t chunk_size = 8 * 1024;
	auto t = [&](size_t initial, size_t to_skip)
	{
		fake_stream s{ initial };
		test(skip(s, to_skip) == std::min(initial, to_skip));
		test(s.m_size == (initial > to_skip ? initial - to_skip : 0));
		test(s.m_calls == std::min(initial, to_skip) / chunk_size + 1);
	};


	t(0, 0);
	t(0, 1);
	t(0, chunk_size);
	t(0, chunk_size + 1);

	t(1, 0);
	t(1, 1);
	t(1, 2);
	t(1, chunk_size);
	t(1, chunk_size + 1);

	t(chunk_size, 0);
	t(chunk_size, 1);
	t(chunk_size, chunk_size - 1);
	t(chunk_size, chunk_size);
	t(chunk_size, chunk_size + 1);

	t(chunk_size + 1, 0);
	t(chunk_size + 1, 1);
	t(chunk_size + 1, chunk_size);
	t(chunk_size + 1, chunk_size + 1);
	t(chunk_size + 1, chunk_size + 2);

	t(chunk_size * 2, 0);
	t(chunk_size * 2, 1);
	t(chunk_size * 2, chunk_size * 2 - 1);
	t(chunk_size * 2, chunk_size * 2);
	t(chunk_size * 2, chunk_size * 2 + 1);

	t(chunk_size * 2 + 1, 0);
	t(chunk_size * 2 + 1, 1);
	t(chunk_size * 2 + 1, chunk_size * 2);
	t(chunk_size * 2 + 1, chunk_size * 2 + 1);
	t(chunk_size * 2 + 1, chunk_size * 2 + 2);
}

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