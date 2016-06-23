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
			test(stream::read<char>(rws.reader) == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws.writer, 'a');
			rws.writer.flush();
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
			test(rws.reader.read_partial_buffer(buffer) == 0);
		});
		std::thread writer([&]
		{
			rws.writer.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_read_empty_buffer)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			test(rws.reader.read_partial_buffer({}) == 0);
			test(stream::read<char>(rws.reader) == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws.writer, 'a');
			rws.writer.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_write_empty_buffer)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			test(stream::read<char>(rws.reader) == 'a');
		});
		std::thread writer([&]
		{
			rws.writer.write_buffer({});
			stream::write(rws.writer, 'a');
			rws.writer.flush();
		});

		reader.join();
		writer.join();
	}

	TEST_CASE(test_reader_buffer_too_small)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([&]
		{
			test(stream::read<char>(rws.reader) == 'h');
			test(stream::read<char>(rws.reader) == 'e');
			test(stream::read<char>(rws.reader) == 'l');
			test(stream::read<char>(rws.reader) == 'l');
			test(stream::read<char>(rws.reader) == 'o');
		});
		std::thread writer([&]
		{
			stream::write(rws.writer, std::array<char, 5>{ 'h', 'e', 'l', 'l', 'o' });
			rws.writer.flush();
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
			test(rws.reader.read_partial_buffer(buffer) == 1);
			test(buffer[0] == 'a');
		});
		std::thread writer([&]
		{
			stream::write(rws.writer, 'a');
			rws.writer.flush();
		});

		reader.join();
		writer.join();
	}
	TEST_CASE(test_writer_throws_after_reader_leave)
	{
		auto rws = stream::create_reader_writer_stream();

		std::thread reader([reader = std::move(rws.reader)]() mutable
		{
			test(stream::read<char>(reader) == 'h');
		});
		std::thread writer([writer = std::move(rws.writer)]() mutable
		{
			expect_exception<stream::reader_writer_stream_closed>([&]
			{
				stream::write(writer, std::array<char, 5>{ 'h', 'e', 'l', 'l', 'o' });
			});
		});

		reader.join();
		writer.join();
	}
}