#include <goldfish/cbor_reader.h>
#include <goldfish/dom_reader.h>
#include <goldfish/stream.h>
#include "unit_test.h"

#include <vector>

namespace goldfish { namespace dom
{
	static uint8_t to_hex(char c)
	{
		if ('0' <= c && c <= '9') return c - '0';
		else if ('a' <= c && c <= 'f') return c - 'a' + 10;
		else if ('A' <= c && c <= 'F') return c - 'A' + 10;
		else throw "Invalid hex character";
	};
	static auto to_vector(const std::string& input)
	{
		std::vector<uint8_t> data;
		for (auto it = input.begin(); it != input.end(); it += 2)
		{
			uint8_t high = to_hex(*it);
			uint8_t low = to_hex(*next(it));
			data.push_back((high << 4) | low);
		}
		return data;
	}
	static auto r(std::string input)
	{
		auto binary = to_vector(input);
		stream::array_ref_reader s(binary);
		auto result = load_in_memory(cbor::read(s));
		test(skip(s, 1) == 0);
		return result;
	};

	TEST_CASE(read_valid_examples)
	{
		test(r("00") == 0ull);
		test(r("01") == 1ull);
		test(r("0a") == 10ull);
		test(r("17") == 23ull);
		test(r("1818") == 24ull);
		test(r("1819") == 25ull);
		test(r("1864") == 100ull);
		test(r("1903e8") == 1000ull);
		test(r("1a000f4240") == 1000000ull);
		test(r("1b000000e8d4a51000") == 1000000000000ull);
		test(r("1bffffffffffffffff") == 18446744073709551615ull);

		test(r("c249010000000000000000") == byte_string(to_vector("010000000000000000")));
		test(r("3b7fffffffffffffff") == -9223372036854775808ll);
		expect_exception<cbor::ill_formatted>([&] { r("3b8000000000000000"); }); // overflow
		test(r("c349010000000000000000") == byte_string(to_vector("010000000000000000")));

		test(r("20") == -1ll);
		test(r("29") == -10ll);
		test(r("3863") == -100ll);
		test(r("3903e7") == -1000ll);
		test(r("3a000f423F") == -1000000ll);

		test(r("f90000") ==  0.0);
		test(r("f98000") == -0.0);
		test(r("f93c00") ==  1.0);
		test(r("fb3ff199999999999a") == 1.1);
		test(r("f93e00") == 1.5);
		test(r("f97bff") == 65504.0);
		test(r("fa47c35000") == 100000.0);
		test(r("fa7f7fffff") == 3.4028234663852886e+38);
		test(r("fb7e37e43c8800759c") == 1.0e+300);
		test(r("f90001") == 5.960464477539063e-8);
		test(r("f90400") == 0.00006103515625);
		test(r("f9c400") == -4.0);
		test(r("fbc010666666666666") == -4.1);
		test(r("f97c00") == std::numeric_limits<double>::infinity());
		test(isnan(r("f97e00").as<double>()));
		test(r("f9fc00") == -std::numeric_limits<double>::infinity());
		test(r("fa7f800000") == std::numeric_limits<double>::infinity());
		test(isnan(r("fa7fc00000").as<double>()));
		test(r("faff800000") == -std::numeric_limits<double>::infinity());
		test(r("fb7ff0000000000000") == std::numeric_limits<double>::infinity());
		test(isnan(r("fb7ff8000000000000").as<double>()));
		test(r("fbfff0000000000000") == -std::numeric_limits<double>::infinity());

		test(r("f4") == false);
		test(r("f5") == true);
		test(r("f6") == nullptr);
		test(r("f7") == undefined{});

		test(r("c074323031332d30332d32315432303a30343a30305a") == text_string("2013-03-21T20:04:00Z"));

		test(r("c11a514b67b0") == 1363896240ull);
		test(r("c1fb41d452d9ec200000") == 1363896240.5);
		test(r("c074323031332d30332d32315432303a30343a30305a") == text_string("2013-03-21T20:04:00Z"));
		test(r("c074323031332d30332d32315432303a30343a30305a") == text_string("2013-03-21T20:04:00Z"));
		test(r("d74401020304") == byte_string(to_vector("01020304")));
		test(r("d818456449455446") == byte_string(to_vector("6449455446")));
		test(r("d82076687474703a2f2f7777772e6578616d706c652e636f6d") == text_string("http://www.example.com"));

		test(r("40") == byte_string(to_vector("")));
		test(r("4401020304") == byte_string(to_vector("01020304")));

		test(r("60") == text_string(""));
		test(r("6161") == text_string("a"));
		test(r("6449455446") == text_string("IETF"));
		test(r("62225c") == text_string("\"\\"));
		test(r("62c3bc") == text_string(u8"\u00fc"));
		test(r("63e6b0b4") == text_string(u8"\u6c34"));
		
		test(r("80") == array{});
		test(r("83010203") == array{ 1ull, 2ull, 3ull });
		test(r("8301820203820405") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull } });
		test(r("98190102030405060708090a0b0c0d0e0f101112131415161718181819") == array{
				1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull,
				10ull, 11ull, 12ull, 13ull, 14ull, 15ull, 16ull,
				17ull, 18ull, 19ull, 20ull, 21ull, 22ull, 23ull,
				24ull, 25ull });

		test(r("a0") == map{});
		test(r("a201020304") == map{ { 1ull, 2ull }, { 3ull, 4ull } });
		test(r("a26161016162820203") == map{
			{ text_string("a"), 1ull },
			{ text_string("b"), array{ 2ull, 3ull } }
		});
		test(r("826161a161626163") == array{ text_string("a"), map{ { text_string("b"), text_string("c") } } });
		test(r("a56161614161626142616361436164614461656145") == map{
			{ text_string("a"), text_string("A") },
			{ text_string("b"), text_string("B") },
			{ text_string("c"), text_string("C") },
			{ text_string("d"), text_string("D") },
			{ text_string("e"), text_string("E") }
		});
		test(r("5f42010243030405ff") == byte_string(to_vector("0102030405")));
		test(r("7f657374726561646d696e67ff") == text_string("streaming"));
		test(r("9fff") == array{});
		test(r("9f018202039f0405ffff") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		test(r("9f01820203820405ff") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		test(r("83018202039f0405ff") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		test(r("83019f0203ff820405") == array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }});
		test(r("9f0102030405060708090a0b0c0d0e0f101112131415161718181819ff") == array{
			1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull,
			9ull, 10ull, 11ull, 12ull, 13ull, 14ull, 15ull,
			16ull, 17ull, 18ull, 19ull, 20ull, 21ull, 22ull,
			23ull, 24ull, 25ull });
		test(r("bf61610161629f0203ffff") == map{
			{ text_string("a"), 1ull },
			{ text_string("b"), array{ 2ull, 3ull } }
		});
		test(r("826161bf61626163ff") == array{ text_string("a"), map{ { text_string("b"), text_string("c") } } });
		test(r("bf6346756ef563416d7421ff") == map{ { text_string("Fun"), true }, { text_string("Amt"), -2ll } });
	}

	TEST_CASE(skip_in_finite_string)
	{
		auto binary = to_vector("6449455446"); // IETF
		stream::array_ref_reader s(binary);
		auto result = cbor::read(s).as<tags::text_string>();

		test(stream::skip(result, 0) == 0);
		test(stream::read<char>(result) == 'I');
		test(stream::skip(result, 1) == 1);
		test(stream::read<char>(result) == 'T');
		test(stream::skip(result, 2) == 1);
	}

	TEST_CASE(skip_in_chunked_string_to_beginning_of_block)
	{
		auto binary = to_vector("7f657374726561646d696e67ff"); // "strea" "ming"
		stream::array_ref_reader s(binary);
		auto result = cbor::read(s).as<tags::text_string>();

		test(stream::skip(result, 0) == 0);
		test(stream::read<char>(result) == 's');
		test(stream::skip(result, 1) == 1);
		test(stream::read<char>(result) == 'r');
		test(stream::skip(result, 2) == 2); // skip to exactly the beginning of a block
		test(stream::read<char>(result) == 'm');
		test(stream::skip(result, 4) == 3);
	}

	TEST_CASE(skip_in_chunked_string_across_block)
	{
		auto binary = to_vector("7f657374726561646d696e67ff"); // "strea" "ming"
		stream::array_ref_reader s(binary);
		auto result = cbor::read(s).as<tags::text_string>();

		test(stream::skip(result, 8) == 8);
		test(stream::read<char>(result) == 'g');
		test(stream::skip(result, 1) == 0);
	}

	TEST_CASE(skip_in_chunked_string_all)
	{
		auto binary = to_vector("7f657374726561646d696e67ff"); // "strea" "ming"
		stream::array_ref_reader s(binary);
		test(stream::skip(cbor::read(s).as<tags::text_string>(), 10) == 9);
	}
}}