#include <goldfish/stream.h>
#include <goldfish/reader_writer_stream.h>
#include <thread>
#include "unit_test.h"

namespace goldfish
{
	TEST_CASE(test_reader_writer_one_byte)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			test(stream::read<char>(rws.first) == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws.second, 'a');
			rws.second.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_reader_writer_empty_stream)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			std::array<byte, 1> buffer;
			test(rws.first.read_buffer(buffer) == 0);
		});
		std::thread writer([&]
		{
			rws.second.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_read_empty_buffer)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			test(rws.first.read_buffer({}) == 0);
			test(stream::read<char>(rws.first) == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws.second, 'a');
			rws.second.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_write_empty_buffer)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			test(stream::read<char>(rws.first) == 'a');
		});
		std::thread writer([&]
		{
			rws.second.write_buffer({});
			stream::write(rws.second, 'a');
			rws.second.flush();
		});

		reader.join();
		writer.join();
	}

	TEST_CASE(test_reader_buffer_too_small)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			test(stream::read<char>(rws.first) == 'h');
			test(stream::read<char>(rws.first) == 'e');
			test(stream::read<char>(rws.first) == 'l');
			test(stream::read<char>(rws.first) == 'l');
			test(stream::read<char>(rws.first) == 'o');
		});
		std::thread writer([&]
		{
			stream::write(rws.second, std::array<char, 5>{ 'h', 'e', 'l', 'l', 'o' });
			rws.second.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_write_buffer_too_small)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			std::array<byte, 5> buffer;
			test(rws.first.read_buffer(buffer) == 1);
			test(buffer[0] == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws.second, 'a');
			rws.second.flush();
		});

		reader.join();
		writer.join();
	}
}