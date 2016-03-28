#include <goldfish/stream.h>
#include "unit_test.h"

namespace goldfish { namespace stream {

static_assert(is_reader<const_buffer_ref_reader>::value, "const_buffer_ref_reader is a reader");
static_assert(!is_reader<vector_writer>::value, "vector_writer is not a reader");
static_assert(!is_writer<const_buffer_ref_reader>::value, "const_buffer_ref_reader is not a writer");
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
		test(seek(s, to_skip) == std::min(initial, to_skip));
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

TEST_CASE(test_copy)
{
	test(copy(read_string("Hello"), string_writer{}).flush() == "Hello");

	{
		string_writer w;
		test(&copy(read_string("Hello"), w) == &w);
		test(w.flush() == "Hello");
	}

	auto test_string_of_size = [](auto size)
	{
		test(copy(read_string(std::string(size, 'a')), string_writer{}).flush() == std::string(size, 'a'));
	};
	test_string_of_size(65535);
	test_string_of_size(65536);
	test_string_of_size(65537);
	test_string_of_size(65536 * 2 - 1);
	test_string_of_size(65536 * 2);
	test_string_of_size(65536 * 2 + 1);
}

}}