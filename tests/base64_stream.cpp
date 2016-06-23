#include <goldfish/base64_stream.h>
#include "unit_test.h"

namespace goldfish { namespace stream
{
	std::string my_base64_encode(const std::string& data)
	{
		auto s = encode_base64_to(string_writer{});
		s.write_buffer({ reinterpret_cast<const byte*>(data.data()), data.size() });
		return s.flush();
	}
	std::string my_base64_decode(const std::string& data)
	{
		return read_all_as_string(decode_base64(read_string_ref(data.c_str())));
	}

	TEST_CASE(base64_encode_0)  { test(my_base64_encode("") == ""); }
	TEST_CASE(base64_encode_1)  { test(my_base64_encode("s") == "cw=="); }
	TEST_CASE(base64_encode_2)  { test(my_base64_encode("su") == "c3U="); }
	TEST_CASE(base64_encode_3)  { test(my_base64_encode("Man") == "TWFu"); }
	TEST_CASE(base64_encode_6)  { test(my_base64_encode("ManMan") == "TWFuTWFu"); }
	TEST_CASE(base64_encode_20) { test(my_base64_encode("any carnal pleasure.") == "YW55IGNhcm5hbCBwbGVhc3VyZS4="); }

	TEST_CASE(base64_decode_0)  { test(my_base64_decode("") == ""); }
	TEST_CASE(base64_decode_1)	{ test(my_base64_decode("cw==") == "s"); }
	TEST_CASE(base64_decode_2)  { test(my_base64_decode("c3U=") == "su"); }
	TEST_CASE(base64_decode_3)  { test(my_base64_decode("TWFu") == "Man"); }
	TEST_CASE(base64_decode_6)  { test(my_base64_decode("TWFuTWFu") == "ManMan"); }
	TEST_CASE(base64_decode_20) { test(my_base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4=") == "any carnal pleasure."); }

	TEST_CASE(base64_decode_0_no_padding) { test(my_base64_decode("") == ""); }
	TEST_CASE(base64_decode_1_no_padding) { test(my_base64_decode("cw") == "s"); }
	TEST_CASE(base64_decode_2_no_padding) { test(my_base64_decode("c3U") == "su"); }
	TEST_CASE(base64_decode_20_no_padding) { test(my_base64_decode("YW55IGNhcm5hbCBwbGVhc3VyZS4") == "any carnal pleasure."); }

	TEST_CASE(base64_decode_wrong_padding_size_1) { expect_exception<ill_formatted_base64_data>([] { my_base64_decode("="); }); }
	TEST_CASE(base64_decode_wrong_padding_size_2) { expect_exception<ill_formatted_base64_data>([] { my_base64_decode("cw="); }); }
	TEST_CASE(base64_decode_wrong_padding_size_3) { expect_exception<ill_formatted_base64_data>([] { my_base64_decode("===="); }); }
	TEST_CASE(base64_padding_in_middle) { expect_exception<ill_formatted_base64_data>([] { my_base64_decode("cw==cw=="); }); }

	TEST_CASE(decode_partial_buffer)
	{
		auto s = decode_base64(read_string("YW55IGNhcm5hbCBwbGVhc3VyZS4"));
		
		// read one at a time (this tests reading 0 and 1 bytes with left overs 0,1,2)
		{
			test(s.read_partial_buffer({}) == 0);
			test(stream::read<char>(s) == 'a');
			test(s.read_partial_buffer({}) == 0);
			test(stream::read<char>(s) == 'n');
			test(s.read_partial_buffer({}) == 0);
			test(stream::read<char>(s) == 'y');
		}

		// read two at a time (this tests reading 2 bytes with left overs 0,2,1)
		{
			byte buffer[2];
			test(s.read_partial_buffer(buffer) == 2 && buffer[0] == ' ' && buffer[1] == 'c');
			test(s.read_partial_buffer(buffer) == 2 && buffer[0] == 'a' && buffer[1] == 'r');
			test(s.read_partial_buffer(buffer) == 2 && buffer[0] == 'n' && buffer[1] == 'a');
		}
		
		// read three at a time (this tests reading 3 bytes with left overs 0,1,2)
		{
			byte buffer[3];
			test(s.read_partial_buffer(buffer) == 3 && buffer[0] == 'l' && buffer[1] == ' ' && buffer[2] == 'p');
			test(stream::read<char>(s) == 'l');
			test(s.read_partial_buffer(buffer) == 3 && buffer[0] == 'e' && buffer[1] == 'a' && buffer[2] == 's');
			test(stream::read<char>(s) == 'u');
			test(s.read_partial_buffer(buffer) == 3 && buffer[0] == 'r' && buffer[1] == 'e' && buffer[2] == '.');
		}
	}

	TEST_CASE(encode_partial_buffer)
	{
		auto s = encode_base64_to(string_writer{});

		// write one at a time (this tests writing 0 and 1 byte with left overs 0,1,2)
		{
			s.write_buffer({});
			stream::write(s, 'a');
			s.write_buffer({});
			stream::write(s, 'n');
			s.write_buffer({});
			stream::write(s, 'y');
		}

		// write two at a time (this tests writing 2 bytes with left overs 0,2,1)
		{
			byte buffer[2];
			buffer[0] = ' '; buffer[1] = 'c'; s.write_buffer(buffer);
			buffer[0] = 'a'; buffer[1] = 'r'; s.write_buffer(buffer);
			buffer[0] = 'n'; buffer[1] = 'a'; s.write_buffer(buffer);
		}

		// write three at a time (this tests writing 3 bytes with left overs)
		{
			byte buffer[3];
			buffer[0] = 'l'; buffer[1] = ' ';  buffer[2] = 'p'; s.write_buffer(buffer);
			stream::write(s, 'l');
			buffer[0] = 'e'; buffer[1] = 'a';  buffer[2] = 's'; s.write_buffer(buffer);
			stream::write(s, 'u');
			buffer[0] = 'r'; buffer[1] = 'e';  buffer[2] = '.'; s.write_buffer(buffer);
		}

		test(s.flush() == "YW55IGNhcm5hbCBwbGVhc3VyZS4=");
	}
}}