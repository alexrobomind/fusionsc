#include "dom.h"
#include <goldfish/json_reader.h>
#include <goldfish/json_writer.h>
#include <goldfish/stream.h>
#include "unit_test.h"

namespace goldfish { namespace dom
{
	TEST_CASE(test_string)
	{
		auto w = [&](const document& d)
		{
			return json::create_writer(stream::string_writer{}).write(d);
		};

		test(w(true) == "true");
		test(w(false) == "false");
		test(w(nullptr) == "null");
		test(w(undefined{}) == "null");
		test(w(0ull) == "0");
		test(w(1ull) == "1");
		test(w(std::numeric_limits<uint64_t>::max()) == "18446744073709551615");
		test(w(0ll) == "0");
		test(w(1ll) == "1");
		test(w(-1ll) == "-1");
		test(w(std::numeric_limits<int64_t>::max()) == "9223372036854775807");
		test(w(std::numeric_limits<int64_t>::min()) == "-9223372036854775808");
		test(w("") == "\"\"");
		test(w(u8"a\u0001\b\n\r\t\"\\/") == "\"a\\u0001\\b\\n\\r\\t\\\"\\\\/\"");
		test(w(array{}) == "[]");
		test(w(array{ 1ull }) == "[1]");
		test(w(array{ 1ull, "abc", array{} }) == "[1,\"abc\",[]]");
		test(w(map{}) == "{}");
		test(w(map{ { "a", 1ull } }) == "{\"a\":1}");
		test(w(map{ { "a", 1ull }, { "b", 2ull } }) == "{\"a\":1,\"b\":2}");

		test(w(std::vector<byte>{}) == "\"\"");
		test(w(std::vector<byte>({ 1 })) == "\"AQ==\"");
		test(w(std::vector<byte>({ 1, 2, 3 })) == "\"AQID\"");
		test(w(map{ { 1ull, 1ull } }) == "{\"1\":1}");
	}

	TEST_CASE(test_roundtrip)
	{
		auto run = [](const char* data)
		{
			test(json::create_writer(stream::string_writer{}).write(json::read(stream::read_string_ref(data))) == data);
		};

		run("[null]");
		run("[true]");
		run("[false]");
		run("[0]");
		run("[\"foo\"]");
		run("[]");
		run("{}");
		run("[0,1]");
		run("{\"foo\":\"bar\"}");
		run("{\"a\":null,\"foo\":\"bar\"}");
		run("[-1]");
		run("[-2147483648]");
		run("[-1234567890123456789]");
		run("[-9223372036854775808]");
		run("[1]");
		run("[2147483647]");
		run("[4294967295]");
		run("[1234567890123456789]");
		run("[9223372036854775807]");
	}

	TEST_CASE(non_string_key)
	{
		auto map = json::create_writer(stream::string_writer{}).start_map();
		map.write(1ull, 1);
		map.write(-1ll, 2);
		map.write(.5, 3);
		map.write(stream::read_string("Key"), 4);
		map.write("Key", 5);

		test(map.flush() == R"({"1":1,"-1":2,"0.500000":3,"S2V5":4,"Key":5})");
	}

	TEST_CASE(test_lossless_floating_point)
	{
		auto run = [](const char* data)
		{
			auto original_float = json::read(stream::read_string_ref(data)).as_double();
			auto round_tripped = json::create_writer(stream::string_writer{}).write(original_float);
			auto new_float = json::read(stream::read_string_ref(round_tripped.c_str())).as_double();
			test(original_float == new_float);
		};
		run("0.0");
		//run("-0.0");
		run("1.2345");
		run("-1.2345");
		run("5e-324");
		//run("2.225073858507201e-308");
		//run("2.2250738585072014e-308");
		//run("1.7976931348623157e308");
	}
}}