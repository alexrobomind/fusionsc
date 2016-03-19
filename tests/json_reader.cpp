#include <goldfish/json_reader.h>
#include <goldfish/dom.h>
#include "unit_test.h"

namespace goldfish { namespace dom
{
	TEST_CASE(json_reader)
	{
		auto r = [&](std::string input)
		{
			stream::const_buffer_ref_reader s({ reinterpret_cast<const byte*>(input.data()), input.size() });
			auto result = load_in_memory(json::read(stream::ref(s)));
			test(seek(s, 1) == 0);
			return result;
		};
		using namespace std::string_literals;

		test(r("\"\"") == "");
		test(r("\"a\"") == "a");
		test(r("\"a\\u0001\\b\\n\\r\\t\\\"\\/\""s) == u8"a\u0001\b\n\r\t\"/");
		test(r("\"\\uD801\\uDC37\""s) == u8"\U00010437");
		test(r("\"\\uE000\""s) == u8"\U0000E000");

		expect_exception<json::ill_formatted_json_data>([&] { r("\"\\uD801\""s); });
		expect_exception<json::ill_formatted_json_data>([&] { r("\"\\uD801a\""s); });
		expect_exception<json::ill_formatted_json_data>([&] { r("\"\\uD801\\n\""s); });
		expect_exception<json::ill_formatted_json_data>([&] { r("\"\\uDC41\\uDC37\""s); });

		test(r("true") == true);
		test(r("false") == false);
		test(r("null") == nullptr);

		test(r("0") == 0ull);
		test(r("1") == 1ull);
		test(r("10") == 10ull);
		test(r("4294967295") == 4294967295ull);
		test(r("18446744073709551615") == 18446744073709551615ull);
		expect_exception<json::integer_overflow_in_json>([&] { r("18446744073709551616"s); });
		expect_exception<json::integer_overflow_in_json>([&] { r("18446744073709551617"s); });
		expect_exception<json::integer_overflow_in_json>([&] { r("18446744073709551618"s); });
		expect_exception<json::integer_overflow_in_json>([&] { r("18446744073709551619"s); });
		//expect_exception<integer_overflow>([&] { r("[00]"s); });

		test(r("-0") == 0ll);
		test(r("-1") == -1ll);
		test(r("-10") == -10ll);
		test(r("-2147483647") == -2147483647ll);
		test(r("-2147483648") == -2147483648ll);
		test(r("-9223372036854775808") == -9223372036854775808ll);
		expect_exception<json::integer_overflow_in_json>([&] { r("-9223372036854775809"s); });

		test(r("0.0") == 0.);
		test(r("0.5") == 0.5);
		test(r("0.50") == 0.5);
		test(r("0.05") == 0.05);
		test(r("50.05") == 50.05);
		test(r("-0.0") == 0.);
		test(r("-0.5") == -0.5);

		test(r("-0.5e1") == -5.);
		test(r("-0.5e01") == -5.);
		test(r("-0.5e001") == -5.);
		test(r("-0.5e+1") == -5.);
		test(r("-0.5E+1") == -5.);
		test(r("-0.5E+10") == -5000000000.);

		test(r("1.7976931348623158e+308") == 1.7976931348623158e+308);
		test(r("2.2204460492503131e-016") == 2.2204460492503131e-016);
		test(r("2.2250738585072014e-308") == 2.2250738585072014e-308);
		test(r("1e309") == std::numeric_limits<double>::infinity());
		test(r("1e-309") == 0.);

		expect_exception<json::ill_formatted_json_data>([&] { r("0."); });
		expect_exception<json::ill_formatted_json_data>([&] { r("[0.]"); });

		test(r("-5E-1") == -0.5);

		test(r("[]") == array{});
		test(r("[ ]") == array{});
		test(r(" [ ]") == array{});

		test(r("[1]") == array{ 1ull });
		test(r("[1,2]") == array{ 1ull, 2ull });
		test(r("[[]]") == array{ dom::document(array{}) });

		test(r("{}") == map{});
		test(r("{\"foo\":1}") == map{ { "foo", 1ull } });
		expect_exception<json::ill_formatted_json_data>([&] { r("{1:1}"); });
	}

	TEST_CASE(json_parse_key)
	{
		// From uint
		test(json::read(stream::read_string_non_owning("{\"1\":1}")).as_map().read_key()->as_int() == 1);
		test(json::read(stream::read_string_non_owning("{\"1\":1}")).as_map().read_key()->as_uint() == 1);
		test(json::read(stream::read_string_non_owning("{\"1\":1}")).as_map().read_key()->as_double() == 1);

		// From int
		test(json::read(stream::read_string_non_owning("{\"-1\":1}")).as_map().read_key()->as_int() == -1);
		test(json::read(stream::read_string_non_owning("{\"-1\":1}")).as_map().read_key()->as_double() == -1.0);

		// From double
		test(json::read(stream::read_string_non_owning("{\"-1.5\":1}")).as_map().read_key()->as_double() == -1.5);

		// As binary
		test(stream::read_all_as_string(json::read(stream::read_string_non_owning("{\"TWFu\":1}")).as_map().read_key()->as_binary()) == "Man");

	}

	struct data_partially_parsed {};

	TEST_CASE(json_failures)
	{
		auto run = [](auto exception, const char* text)
		{
			auto s = stream::read_string_non_owning(text);
			expect_exception<decltype(exception)>([&]
			{
				dom::load_in_memory(json::read(stream::ref(s)));
				if (seek(s, 1) == 1)
					throw data_partially_parsed{};
			});
		};
		run(stream::unexpected_end_of_stream{}, "[\"Unclosed array\"");
		run(json::ill_formatted_json_data{}, "{unquoted_key: \"keys must be quoted\"}");
		run(json::ill_formatted_json_data{}, "[\"extra comma\",]");
		run(json::ill_formatted_json_data{}, "[\"double extra comma\",,]");
		run(json::ill_formatted_json_data{}, "[   , \"<--missing value\"]");
		run(data_partially_parsed{}, "[\"Comma after the close\"],");
		run(data_partially_parsed{}, "[\"Extra close\"]]");
		run(json::ill_formatted_json_data{}, "{\"Extra comma\": true, }");
		run(data_partially_parsed{}, "{\"Extra value after close\": true} \"misplaced quoted value\"");
		run(json::ill_formatted_json_data{}, "{\"Illegal expression\": 1 + 2}");
		run(json::ill_formatted_json_data{}, "{\"Illegal invocation\": alert()}");
		run(json::ill_formatted_json_data{}, "{\"Numbers cannot have leading zeroes\": 013}");
		run(json::ill_formatted_json_data{}, "{\"Numbers cannot be hex\": 0x14}");
		run(json::ill_formatted_json_data{}, "[\"Illegal backslash escape : \x15\"]");
		run(json::ill_formatted_json_data{}, "[\naked]");
		run(json::ill_formatted_json_data{}, "[\"Illegal backslash escape : \017\"]");
		run(json::ill_formatted_json_data{}, "{\"Missing colon\" null}");
		run(json::ill_formatted_json_data{}, "{\"Double colon\":: null}");
		run(json::ill_formatted_json_data{}, "{\"Comma instead of colon\", null}");
		run(json::ill_formatted_json_data{}, "[\"Colon instead of comma\": false]");
		run(json::ill_formatted_json_data{}, "[\"Bad value\", truth]");
		run(json::ill_formatted_json_data{}, "['single quote']");
		run(json::ill_formatted_json_data{}, "[\"\ttab\tcharacter\tin\tstring\t\"]");
		run(json::ill_formatted_json_data{}, "[\"tab\\   character\\   in\\  string\\  \"]");
		run(json::ill_formatted_json_data{}, "[\"line\nbreak\"]");
		run(json::ill_formatted_json_data{}, "[\"line\\\nbreak\"]");
		run(json::ill_formatted_json_data{}, "[0e]");
		run(json::ill_formatted_json_data{}, "[0e+]");
		run(json::ill_formatted_json_data{}, "[0e+-1]");
		run(stream::unexpected_end_of_stream{}, "{\"Comma instead if closing brace\": true,");
		run(json::ill_formatted_json_data{}, "[\"mismatch\"}");
	}

	TEST_CASE(test_success)
	{
		auto run = [](const char* text)
		{
			auto s = stream::read_string_non_owning(text);
			dom::load_in_memory(json::read(stream::ref(s)));
			test(json::details::peek_non_space(s) == nullopt);
		};

		run(R"json([
    "JSON Test Pattern pass1",
    {"object with 1 member":["array with 1 element"]},
    {},
    [],
    -42,
    true,
    false,
    null,
    {
        "integer": 1234567890,
        "real": -9876.543210,
        "e": 0.123456789e-12,
        "E": 1.234567890E+34,
        "":  23456789012E66,
        "zero": 0,
        "one": 1,
        "space": " ",
        "quote": "\"",
        "backslash": "\\",
        "controls": "\b\f\n\r\t",
        "slash": "/ & \/",
        "alpha": "abcdefghijklmnopqrstuvwyz",
        "ALPHA": "ABCDEFGHIJKLMNOPQRSTUVWYZ",
        "digit": "0123456789",
        "0123456789": "digit",
        "special": "`1~!@#$%^&*()_+-={':[,]}|;.</>?",
        "hex": "\u0123\u4567\u89AB\uCDEF\uabcd\uef4A",
        "true": true,
        "false": false,
        "null": null,
        "array":[  ],
        "object":{  },
        "address": "50 St. James Street",
        "url": "http://www.JSON.org/",
        "comment": "// /* <!-- --",
        "# -- --> */": " ",
        " s p a c e d " :[1,2 , 3

			,

			4 , 5        ,          6           ,7        ],"compact":[1,2,3,4,5,6,7],
        "jsontext": "{\"object with 1 member\":[\"array with 1 element\"]}",
        "quotes": "&#34; \u0022 %22 0x22 034 &#x22;",
        "\/\\\"\uCAFE\uBABE\uAB98\uFCDE\ubcda\uef4A\b\f\n\r\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?"
: "A key can be any string"
    },
    0.5 ,98.6
,
99.44
,

1066,
1e1,
0.1e1,
1e-1,
1e00,2e+00,2e-00
,"rosebud"])json");
		run(R"json([[[[[[[[[[[[[[[[[[["Not too deep"]]]]]]]]]]]]]]]]]]])json");
		run(R"json({
    "JSON Test Pattern pass3": {
        "The outermost value": "must be an object or array.",
        "In this test": "It is an object."
    }
}
)json");
	}

}}