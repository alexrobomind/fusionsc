#include <gold_fish/dom_reader.h>
#include <gold_fish/dom_writer.h>
#include <gold_fish/json_reader.h>
#include <gold_fish/json_writer.h>
#include <gold_fish/stream.h>
#include "unit_test.h"

namespace gold_fish { namespace dom
{
	TEST_CASE(string)
	{
		auto w = [&](const document& d)
		{
			stream::string_writer s;
			write(json::write(s), d);
			s.flush();
			return s.data;
		};

		TEST(w(true) == "true");
		TEST(w(false) == "false");
		TEST(w(nullptr) == "null");
		TEST(w(undefined{}) == "null");
		TEST(w(0ull) == "0");
		TEST(w(1ull) == "1");
		TEST(w(std::numeric_limits<uint64_t>::max()) == "18446744073709551615");
		TEST(w(0ll) == "0");
		TEST(w(1ll) == "1");
		TEST(w(-1ll) == "-1");
		TEST(w(std::numeric_limits<int64_t>::max()) == "9223372036854775807");
		TEST(w(std::numeric_limits<int64_t>::min()) == "-9223372036854775808");
		TEST(w(text_string("")) == "\"\"");
		TEST(w(text_string(u8"a\u0001\b\n\r\t\"\\/")) == "\"a\\u0001\\b\\n\\r\\t\\\"\\\\/\"");
		TEST(w(array{}) == "[]");
		TEST(w(array{ 1ull }) == "[1]");
		TEST(w(array{ 1ull, text_string("abc"), array{} }) == "[1,\"abc\",[]]");
		TEST(w(map{}) == "{}");
		TEST(w(map{ { text_string("a"), 1ull } }) == "{\"a\":1}");
		TEST(w(map{ { text_string("a"), 1ull }, { text_string("b"), 2ull } }) == "{\"a\":1,\"b\":2}");

		TEST(w(text_string("/B")) == "\"/B\"");
		TEST(w(byte_string({})) == "\"\\/B\"");
		TEST(w(byte_string({ 1 })) == "\"\\/BAQ==\"");
		TEST(w(byte_string({ 1, 2, 3 })) == "\"\\/BAQID\"");
		TEST(w(map{ { 1ull, 1ull } }) == "{1:1}");
	}

	TEST_CASE(test_roundtrip)
	{
		auto run = [](const char* data)
		{
			auto w = stream::string_writer{};
			auto r = stream::read_string_literal(data);
			dom::write(json::write(w), dom::load_in_memory(json::read(r)));
			w.flush();
			TEST(data == w.data);
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
		//run("[0.0]");
		//run("[-0.0]");
		run("[1.2345]");
		run("[-1.2345]");
		//run("[5e-324]");
		//run("[2.225073858507201e-308]");
		//run("[2.2250738585072014e-308]");
		//run("[1.7976931348623157e308]");
	}
}}