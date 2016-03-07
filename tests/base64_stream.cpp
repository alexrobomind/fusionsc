#include <gold_fish/base64_stream.h>
#include "unit_test.h"

namespace gold_fish { namespace stream
{
	std::string my_base64_encode(const std::string& data)
	{
		string_writer writer;
		auto s = base64(buffer<4>(ref(writer)));
		s.write_buffer({ reinterpret_cast<const uint8_t*>(data.data()), data.size() });
		s.flush();
		return std::move(writer.data);
	}
	std::string my_base64_decode(const std::string& data)
	{
		return read_all_as_string(base64(buffer<4>(read_string_literal(data.c_str()))));
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
}}