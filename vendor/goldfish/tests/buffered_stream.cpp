#include <goldfish/buffered_stream.h>

#include "unit_test.h"

namespace goldfish { namespace stream
{
	TEST_CASE(test_buffered_reader_single_character_read)
	{
		auto s = buffer<3>(read_string("abcd"));
		test(s.peek<char>() == 'a'); test(stream::read<char>(s) == 'a');
		test(s.peek<char>() == 'b'); test(stream::read<char>(s) == 'b'); test(s.peek<std::array<char, 3>>() == nullopt);
		test(s.peek<char>() == 'c'); test(stream::read<char>(s) == 'c'); test(s.peek<std::array<char, 3>>() == nullopt);
		test(s.peek<char>() == 'd'); test(stream::read<char>(s) == 'd'); test(s.peek<std::array<char, 3>>() == nullopt);
		test(s.peek<char>() == nullopt);
	}
	TEST_CASE(test_buffered_reader_two_character_peek_and_read)
	{
		auto s = buffer<3>(read_string("abcdef"));
		test(s.peek<std::array<char, 2>>() == std::array<char, 2>{ 'a', 'b' }); test(stream::read<std::array<char, 2>>(s) == std::array<char, 2>{ 'a', 'b' });
		test(s.peek<std::array<char, 2>>() == std::array<char, 2>{ 'c', 'd' }); test(stream::read<std::array<char, 2>>(s) == std::array<char, 2>{ 'c', 'd' });
		test(s.peek<std::array<char, 2>>() == std::array<char, 2>{ 'e', 'f' }); test(stream::read<std::array<char, 2>>(s) == std::array<char, 2>{ 'e', 'f' });
	}
	TEST_CASE(test_buffered_reader_two_character_read)
	{
		auto s = buffer<3>(read_string("abcdef"));
		test(stream::read<std::array<char, 2>>(s) == std::array<char, 2>{ 'a', 'b' });
		test(stream::read<std::array<char, 2>>(s) == std::array<char, 2>{ 'c', 'd' });
		test(stream::read<std::array<char, 2>>(s) == std::array<char, 2>{ 'e', 'f' });
	}
	TEST_CASE(test_buffered_reader_three_character_read)
	{
		auto s = buffer<3>(read_string("abcdefghijk"));
		test(s.peek<std::array<char, 3>>() == std::array<char, 3>{ 'a', 'b', 'c' }); test(stream::read<std::array<char, 3>>(s) == std::array<char, 3>{ 'a', 'b', 'c' });
		test(stream::read<char>(s) == 'd');
		test(s.peek<std::array<char, 3>>() == std::array<char, 3>{ 'e', 'f', 'g' }); test(stream::read<std::array<char, 3>>(s) == std::array<char, 3>{ 'e', 'f', 'g' });
		test(stream::read<char>(s) == 'h');
		test(s.peek<std::array<char, 3>>() == std::array<char, 3>{ 'i', 'j', 'k' }); test(stream::read<std::array<char, 3>>(s) == std::array<char, 3>{ 'i', 'j', 'k' });
	}
	TEST_CASE(test_buffered_reader_four_character_read)
	{
		auto s = buffer<3>(read_string("abcdefghijkl"));
		test(stream::read<std::array<char, 4>>(s) == std::array<char, 4>{ 'a', 'b', 'c', 'd' });
		test(stream::read<std::array<char, 4>>(s) == std::array<char, 4>{ 'e', 'f', 'g', 'h' });
		test(stream::read<std::array<char, 4>>(s) == std::array<char, 4>{ 'i', 'j', 'k', 'l' });
	}
	TEST_CASE(test_buffered_reader_read_buffer)
	{
		auto s = buffer<3>(read_string("abcdefghijkl"));
		{
			std::array<byte, 1> buffer;
			test(s.read_partial_buffer(buffer) == 1);
			test(buffer == std::array<uint8_t, 1>{'a'});
		}
		{
			std::array<byte, 2> buffer;
			test(s.read_partial_buffer(buffer) == 2);
			test(buffer == std::array<byte, 2>{'b', 'c'});
		}
		{
			std::array<byte, 3> buffer;
			test(s.read_partial_buffer(buffer) == 3);
			test(buffer == std::array<byte, 3>{'d', 'e', 'f'});
		}
		{
			std::array<byte, 4> buffer = { 'X', 'X', 'X', 'X' };
			test(s.read_partial_buffer(buffer) == 3);
			test(buffer == std::array<byte, 4>{'g', 'h', 'i', 'X' });

			test(s.read_partial_buffer(buffer) == 3);
			test(buffer == std::array<byte, 4>{'j', 'k', 'l', 'X' });
		}
	}
	TEST_CASE(test_buffered_seek)
	{
		auto s = buffer<3>(read_string("abcdef"));
		test(stream::seek(s, 1) == 1);
		test(s.peek<char>() == 'b');
		test(stream::seek(s, 1) == 1);
		test(s.peek<char>() == 'c');
		test(stream::seek(s, 5) == 4);
		test(stream::seek(s, 1) == 0);
	}
	TEST_CASE(test_move_buffered_reader)
	{
		auto s = buffer<3>(read_string("abcdef"));
		test(stream::read<char>(s) == 'a');
		auto t = std::move(s);
		test(stream::read<char>(t) == 'b');
	}
	TEST_CASE(test_buffered_writer)
	{
		vector_writer x;
		auto stream = buffer<2>(ref(x));
		stream.write<byte>(1);
		test(x.data().empty());

		stream.write<byte>(2);
		test(x.data().empty());

		stream.flush();

		test(x.data() == std::vector<byte>{1, 2});
	}
}}