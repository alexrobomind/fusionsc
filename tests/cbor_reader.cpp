#include "cbor_reader.h"
#include "dom_reader.h"
#include "unit_test.h"
#include "stream.h"

#include <vector>

namespace gold_fish { namespace dom
{
	TEST_CASE(read_valid_examples)
	{
		auto to_hex = [](char c)
		{
			     if ('0' <= c && c <= '9') return c - '0';
			else if ('a' <= c && c <= 'f') return c - 'a' + 10;
			else if ('A' <= c && c <= 'F') return c - 'A' + 10;
			else throw "Invalid hex character";
		};
		auto to_vector = [&](const std::string& input) -> std::vector<uint8_t>
		{
			std::vector<uint8_t> data;
			for (auto it = input.begin(); it != input.end(); it += 2)
			{
				uint8_t high = to_hex(*it);
				uint8_t low = to_hex(*next(it));
				data.push_back((high << 4) | low);
			}
			return data;
		};
		auto r = [&](std::string input)
		{
			auto binary = to_vector(input);
			stream::array_ref_reader s(array_ref<const uint8_t>{ binary });
			auto result = load_in_memory(cbor::read(stream::ref(s)));
			TEST(skip(s, 1) == 0);
			return result;
		};

		TEST(r("00") == 0ull);
		TEST(r("01") == 1ull);
		TEST(r("0a") == 10ull);
		TEST(r("17") == 23ull);
		TEST(r("1818") == 24ull);
		TEST(r("1819") == 25ull);
		TEST(r("1864") == 100ull);
		TEST(r("1903e8") == 1000ull);
		TEST(r("1a000f4240") == 1000000ull);
		TEST(r("1b000000e8d4a51000") == 1000000000000ull);
		TEST(r("1bffffffffffffffff") == 18446744073709551615ull);

		TEST(r("c249010000000000000000") == byte_string(to_vector("010000000000000000")));
		TEST(r("3b7fffffffffffffff") == -9223372036854775808ll);
		expect_exception<cbor::ill_formatted>([&] { r("3b8000000000000000"); }); // overflow
		TEST(r("c349010000000000000000") == byte_string(to_vector("010000000000000000")));

		TEST(r("20") == -1ll);
		TEST(r("29") == -10ll);
		TEST(r("3863") == -100ll);
		TEST(r("3903e7") == -1000ll);

		TEST(r("f90000") ==  0.0);
		TEST(r("f98000") == -0.0);
		TEST(r("f93c00") ==  1.0);
		TEST(r("fb3ff199999999999a") == 1.1);
		TEST(r("f93e00") == 1.5);
		TEST(r("f97bff") == 65504.0);
		TEST(r("fa47c35000") == 100000.0);
		TEST(r("fa7f7fffff") == 3.4028234663852886e+38);
		TEST(r("fb7e37e43c8800759c") == 1.0e+300);
		TEST(r("f90001") == 5.960464477539063e-8);
		TEST(r("f90400") == 0.00006103515625);
		TEST(r("f9c400") == -4.0);
		TEST(r("fbc010666666666666") == -4.1);
		TEST(r("f97c00") == std::numeric_limits<double>::infinity());
		TEST(isnan(r("f97e00").as<double>()));
		TEST(r("f9fc00") == -std::numeric_limits<double>::infinity());
		TEST(r("fa7f800000") == std::numeric_limits<double>::infinity());
		TEST(isnan(r("fa7fc00000").as<double>()));
		TEST(r("faff800000") == -std::numeric_limits<double>::infinity());
		TEST(r("fb7ff0000000000000") == std::numeric_limits<double>::infinity());
		TEST(isnan(r("fb7ff8000000000000").as<double>()));
		TEST(r("fbfff0000000000000") == -std::numeric_limits<double>::infinity());

		TEST(r("f4") == false);
		TEST(r("f5") == true);
		TEST(r("f6") == nullptr);
		TEST(r("f7") == undefined{});

		TEST(r("c074323031332d30332d32315432303a30343a30305a") == text_string("2013-03-21T20:04:00Z"));

		TEST(r("c11a514b67b0") == 1363896240ull);
		TEST(r("c1fb41d452d9ec200000") == 1363896240.5);
		TEST(r("c074323031332d30332d32315432303a30343a30305a") == text_string("2013-03-21T20:04:00Z"));
		TEST(r("c074323031332d30332d32315432303a30343a30305a") == text_string("2013-03-21T20:04:00Z"));
		TEST(r("d74401020304") == byte_string(to_vector("01020304")));
		TEST(r("d818456449455446") == byte_string(to_vector("6449455446")));
		TEST(r("d82076687474703a2f2f7777772e6578616d706c652e636f6d") == text_string("http://www.example.com"));

		TEST(r("40") == byte_string(to_vector("")));
		TEST(r("4401020304") == byte_string(to_vector("01020304")));

		TEST(r("60") == text_string(""));
		TEST(r("6161") == text_string("a"));
		TEST(r("6449455446") == text_string("IETF"));
		TEST(r("62225c") == text_string("\"\\"));
		TEST(r("62c3bc") == text_string(u8"\u00fc"));
		TEST(r("63e6b0b4") == text_string(u8"\u6c34"));
		
		TEST(r("80") == array{});
		TEST(r("83010203") == array{ 1ull, 2ull, 3ull });
		TEST(r("8301820203820405") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull } });
		TEST(r("98190102030405060708090a0b0c0d0e0f101112131415161718181819") == array{
				1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull,
				10ull, 11ull, 12ull, 13ull, 14ull, 15ull, 16ull,
				17ull, 18ull, 19ull, 20ull, 21ull, 22ull, 23ull,
				24ull, 25ull });

		TEST(r("a0") == map{});
		TEST(r("a201020304") == map{ { 1ull, 2ull }, { 3ull, 4ull } });
		TEST(r("a26161016162820203") == map{
			{ text_string("a"), 1ull },
			{ text_string("b"), array{ 2ull, 3ull } }
		});
		TEST(r("826161a161626163") == array{ text_string("a"), map{ { text_string("b"), text_string("c") } } });
		TEST(r("a56161614161626142616361436164614461656145") == map{
			{ text_string("a"), text_string("A") },
			{ text_string("b"), text_string("B") },
			{ text_string("c"), text_string("C") },
			{ text_string("d"), text_string("D") },
			{ text_string("e"), text_string("E") }
		});
		TEST(r("5f42010243030405ff") == byte_string(to_vector("0102030405")));
		TEST(r("7f657374726561646d696e67ff") == text_string("streaming"));
		TEST(r("9fff") == array{});
		TEST(r("9f018202039f0405ffff") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		TEST(r("9f01820203820405ff") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		TEST(r("83018202039f0405ff") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		TEST(r("83019f0203ff820405") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		TEST(r("9f0102030405060708090a0b0c0d0e0f101112131415161718181819ff") == array{
			1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull,
			9ull, 10ull, 11ull, 12ull, 13ull, 14ull, 15ull,
			16ull, 17ull, 18ull, 19ull, 20ull, 21ull, 22ull,
			23ull, 24ull, 25ull });
		TEST(r("bf61610161629f0203ffff") == map{
			{ text_string("a"), 1ull },
			{ text_string("b"), array{ 2ull, 3ull } }
		});
		TEST(r("826161bf61626163ff") == array{ text_string("a"), map{ { text_string("b"), text_string("c") } } });
		TEST(r("bf6346756ef563416d7421ff") == map{ { text_string("Fun"), true }, { text_string("Amt"), -2ll } });
	}
}}