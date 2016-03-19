#include <goldfish/stream.h>
#include <goldfish/reader_writer_stream.h>
#include <thread>
#include "unit_test.h"

namespace goldfish
{
	TEST_CASE(test_reader_writer_one_byte)
	{
		stream::reader_writer_stream rws;

		std::thread reader([&]
		{
			test(stream::read<char>(rws) == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws, 'a');
			rws.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_reader_writer_empty_stream)
	{
		stream::reader_writer_stream rws;

		std::thread reader([&]
		{
			std::array<byte, 1> buffer;
			test(rws.read_buffer(buffer) == 0);
		});
		std::thread writer([&]
		{
			rws.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_read_empty_buffer)
	{
		stream::reader_writer_stream rws;

		std::thread reader([&]
		{
			test(rws.read_buffer({}) == 0);
			test(stream::read<char>(rws) == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws, 'a');
			rws.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_write_empty_buffer)
	{
		stream::reader_writer_stream rws;

		std::thread reader([&]
		{
			test(stream::read<char>(rws) == 'a');
		});
		std::thread writer([&]
		{
			rws.write_buffer({});
			stream::write(rws, 'a');
			rws.flush();
		});

		reader.join();
		writer.join();
	}

	TEST_CASE(test_reader_buffer_too_small)
	{
		stream::reader_writer_stream rws;

		std::thread reader([&]
		{
			test(stream::read<char>(rws) == 'h');
			test(stream::read<char>(rws) == 'e');
			test(stream::read<char>(rws) == 'l');
			test(stream::read<char>(rws) == 'l');
			test(stream::read<char>(rws) == 'o');
		});
		std::thread writer([&]
		{
			stream::write(rws, std::array<char, 5>{ 'h', 'e', 'l', 'l', 'o' });
			rws.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_write_buffer_too_small)
	{
		stream::reader_writer_stream rws;

		std::thread reader([&]
		{
			std::array<byte, 5> buffer;
			test(rws.read_buffer(buffer) == 1);
			test(buffer[0] == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws, 'a');
			rws.flush();
		});

		reader.join();
		writer.join();
	}
}