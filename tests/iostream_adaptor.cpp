#include <goldfish/iostream_adaptor.h>
#include <sstream>
#include "unit_test.h"

namespace goldfish
{
	TEST_CASE(istream_reader)
	{
		std::stringstream s("Hello");
		test(stream::read_all_as_string(stream::istream_reader_ref{ s }) == "Hello");
	}
	TEST_CASE(ostream_writer)
	{
		std::stringstream s;
		stream::ostream_writer_ref writer{ s };
		stream::copy(stream::read_string("Hello"), writer);
		writer.flush();
		test(s.str() == "Hello");
	}
	TEST_CASE(test_create_istream)
	{
		auto s = stream::make_istream<4>(stream::read_string("Hello world"));

		std::string word;
		s >> word;
		test(word == "Hello");

		s >> word;
		test(word == "world");
	}
}