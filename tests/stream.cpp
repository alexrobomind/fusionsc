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

		size_t read_partial_buffer(buffer_ref buffer)
		{
			// pretend we filled buffer with data
			auto cb = std::min(buffer.size(), m_size);
			m_size -= cb;
			return cb;
		}
	};
	static const size_t chunk_size = 8 * 1024;
	auto t = [&](size_t initial, size_t to_skip)
	{
		fake_stream s { initial };
		test(seek(s, to_skip) == std::min(initial, to_skip));
		test(s.m_size == (initial > to_skip ? initial - to_skip : 0));
	};

	t(0, 0);
	t(0, 1);
	t(0, typical_buffer_length);
	t(0, typical_buffer_length + 1);

	t(1, 0);
	t(1, 1);
	t(1, 2);
	t(1, typical_buffer_length);
	t(1, typical_buffer_length + 1);

	t(typical_buffer_length, 0);
	t(typical_buffer_length, 1);
	t(typical_buffer_length, typical_buffer_length - 1);
	t(typical_buffer_length, typical_buffer_length);
	t(typical_buffer_length, typical_buffer_length + 1);

	t(typical_buffer_length + 1, 0);
	t(typical_buffer_length + 1, 1);
	t(typical_buffer_length + 1, typical_buffer_length);
	t(typical_buffer_length + 1, typical_buffer_length + 1);
	t(typical_buffer_length + 1, typical_buffer_length + 2);

	t(typical_buffer_length * 2, 0);
	t(typical_buffer_length * 2, 1);
	t(typical_buffer_length * 2, typical_buffer_length * 2 - 1);
	t(typical_buffer_length * 2, typical_buffer_length * 2);
	t(typical_buffer_length * 2, typical_buffer_length * 2 + 1);

	t(typical_buffer_length * 2 + 1, 0);
	t(typical_buffer_length * 2 + 1, 1);
	t(typical_buffer_length * 2 + 1, typical_buffer_length * 2);
	t(typical_buffer_length * 2 + 1, typical_buffer_length * 2 + 1);
	t(typical_buffer_length * 2 + 1, typical_buffer_length * 2 + 2);
}

TEST_CASE(test_copy)
{
	{
		string_writer w;
		copy(read_string("Hello"), w);
		test(w.flush() == "Hello");
	}

	auto test_string_of_size = [](auto size)
	{
		string_writer w;
		copy(read_string(std::string(size, 'a')), w);
		test(w.flush() == std::string(size, 'a'));
	};
	test_string_of_size(typical_buffer_length - 1);
	test_string_of_size(typical_buffer_length);
	test_string_of_size(typical_buffer_length + 1);
	test_string_of_size(typical_buffer_length * 2 - 1);
	test_string_of_size(typical_buffer_length * 2);
	test_string_of_size(typical_buffer_length * 2 + 1);
}

}}