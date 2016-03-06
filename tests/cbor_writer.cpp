#include <gold_fish/dom_writer.h>
#include <gold_fish/cbor_writer.h>
#include <gold_fish/stream.h>
#include "unit_test.h"

namespace gold_fish { namespace dom
{
	struct in_memory_write_stream
	{
		void write(uint8_t x) { data.push_back(x); }
		template <class T> void write(T x) { write(&x, sizeof(x)); }
		void write(const void* buffer, size_t size)
		{
			data.insert(data.end(), reinterpret_cast<const char*>(buffer), reinterpret_cast<const char*>(buffer) + size);
		}
		std::vector<uint8_t> data;
	};

	static std::string to_hex_string(const std::vector<uint8_t>& data)
	{
		std::string result;
		for (auto&& x : data)
		{
			result += "0123456789abcdef"[x >> 4];
			result += "0123456789abcdef"[x & 0b1111];
		}
		return result;
	}

	TEST_CASE(write_valid_examples)
	{
		auto to_hex = [](char c) -> uint8_t
		{
			if ('0' <= c && c <= '9') return c - '0';
			else if ('a' <= c && c <= 'f') return c - 'a' + 10;
			else if ('A' <= c && c <= 'F') return c - 'A' + 10;
			else throw "Invalid hex character";
		};
		auto to_vector = [&](const std::string& input)
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
		auto w = [&](const document& d)
		{
			stream::vector_writer s;
			write(cbor::write(s), d);
			s.flush();
			return to_hex_string(s.data);
		};
		TEST(w(0ull) == "00");
		TEST(w(1ull) == "01");
		TEST(w(10ull) == "0a");
		TEST(w(23ull) == "17");
		TEST(w(24ull) == "1818");
		TEST(w(25ull) == "1819");
		TEST(w(100ull) == "1864");
		TEST(w(1000ull) == "1903e8");
		TEST(w(1000000ull) == "1a000f4240");
		TEST(w(1000000000000ull) == "1b000000e8d4a51000");
		TEST(w(18446744073709551615ull) == "1bffffffffffffffff");

		//TEST(w(tagged{ 2, to_vector("010000000000000000") }) == "c249010000000000000000");
		TEST(w(-18446744073709551615ll) == "3bfffffffffffffffe");
		//TEST(w(tagged{ 3, to_vector("010000000000000000") }) == "c349010000000000000000");

		TEST(w(-1ll) == "20");
		TEST(w(-10ll) == "29");
		TEST(w(-100ll) == "3863");
		TEST(w(-1000ll) == "3903e7");

		//TEST(w(0.0) == "f90000");
		//TEST(w(-0.0) == "f98000");
		//TEST(w(1.0) == "f93c00");
		//TEST(w(1.1) == "fb3ff199999999999a");
		//TEST(w(1.5) == "f93e00");
		//TEST(w(65504.0) == "f97bff");
		//TEST(w(100000.0) == "fa47c35000");
		//TEST(w(3.4028234663852886e+38) == "fa7f7fffff");
		//TEST(w(1.0e+300) == "fb7e37e43c8800759c");
		//TEST(w(5.960464477539063e-8) == "f90001");
		//TEST(w(0.00006103515625) == "f90400");
		//TEST(w(-4.0) == "f9c400");
		//TEST(w(-4.1) == "fbc010666666666666");
		//TEST(w(std::numeric_limits<double>::infinity()) == "f97c00");
		//TEST(isnan(r("f97e00").as<double>()));
		//TEST(w(-std::numeric_limits<double>::infinity()) == "f9fc00");
		//TEST(w(std::numeric_limits<double>::infinity()) == "fa7f800000");
		//TEST(isnan(r("fa7fc00000").as<double>()));
		//TEST(w(-std::numeric_limits<double>::infinity()) == "faff800000");
		//TEST(w(std::numeric_limits<double>::infinity()) == "fb7ff0000000000000");
		//TEST(isnan(r("fb7ff8000000000000").as<double>()));
		//TEST(w(-std::numeric_limits<double>::infinity()) == "fbfff0000000000000");

		TEST(w(false) == "f4");
		TEST(w(true) == "f5");
		TEST(w(nullptr) == "f6");
		TEST(w(undefined{}) == "f7");

		//TEST(w(tagged{ 0, "2013-03-21T20:04:00Z"s }) == "c074323031332d30332d32315432303a30343a30305a");

		//TEST(w(tagged{ 1, 1363896240ull }) == "c11a514b67b0");
		//TEST(w(tagged{ 1, 1363896240.5 }) == "c1fb41d452d9ec200000");
		//TEST(w(tagged{ 0, "2013-03-21T20:04:00Z"s }) == "c074323031332d30332d32315432303a30343a30305a");
		//TEST(w(tagged{ 0, "2013-03-21T20:04:00Z"s }) == "c074323031332d30332d32315432303a30343a30305a");
		//TEST(w(tagged{ 23, to_vector("01020304") }) == "d74401020304");
		//TEST(w(tagged{ 24, to_vector("6449455446") }) == "d818456449455446");
		//TEST(w(tagged{ 32, "http://www.example.com"s }) == "d82076687474703a2f2f7777772e6578616d706c652e636f6d");

		TEST(w(byte_string(to_vector(""))) == "40");
		TEST(w(byte_string(to_vector("01020304"))) == "4401020304");

		TEST(w(text_string("")) == "60");
		TEST(w(text_string("a")) == "6161");
		TEST(w(text_string("IETF")) == "6449455446");
		TEST(w(text_string("\"\\")) == "62225c");
		TEST(w(text_string(u8"\u00fc")) == "62c3bc");
		TEST(w(text_string(u8"\u6c34")) == "63e6b0b4");

		TEST(w(array{}) == "80");
		TEST(w(array{ 1ull, 2ull, 3ull }) == "83010203");
		TEST(w(array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull } }) == "8301820203820405");
		TEST(w(array{
			1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull,
				10ull, 11ull, 12ull, 13ull, 14ull, 15ull, 16ull,
				17ull, 18ull, 19ull, 20ull, 21ull, 22ull, 23ull,
				24ull, 25ull }) == "98190102030405060708090a0b0c0d0e0f101112131415161718181819");

		TEST(w(map{}) == "a0");
		TEST(w(map{ { 1ull, 2ull },{ 3ull, 4ull } }) == "a201020304");
		TEST(w(map{
			{ text_string("a"), 1ull },
			{ text_string("b"), array{ 2ull, 3ull } }
		}) == "a26161016162820203");
		TEST(w(array{ text_string("a"), map{ { text_string("b"), text_string("c") } } }) == "826161a161626163");
		TEST(w(map{
			{ text_string("a"), text_string("A") },
			{ text_string("b"), text_string("B") },
			{ text_string("c"), text_string("C") },
			{ text_string("d"), text_string("D") },
			{ text_string("e"), text_string("E") }
		}) == "a56161614161626142616361436164614461656145");

		// input used indefinite array
		//
		//TEST(w(byte_string(to_vector("0102030405"))) == "5f42010243030405ff");
		//TEST(w("streaming"s) == "7f657374726561646d696e67ff");   
		//TEST(w(array{}) == "9fff");                              
		//TEST(w(array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }}) == "9f018202039f0405ffff");
		//TEST(w(array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }}) == "9f01820203820405ff");
		//TEST(w(array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }}) == "83018202039f0405ff");
		//TEST(w(array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull }}) == "83019f0203ff820405");
		//TEST(w(array{
		//	1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull,
		//	9ull, 10ull, 11ull, 12ull, 13ull, 14ull, 15ull,
		//	16ull, 17ull, 18ull, 19ull, 20ull, 21ull, 22ull,
		//	23ull, 24ull, 25ull }) == "9f0102030405060708090a0b0c0d0e0f101112131415161718181819ff");
		//TEST(w(map{
		//	{ "a"s, 1ull },
		//	{ "b"s, array{ 2ull, 3ull } }
		//}) == "bf61610161629f0203ffff");
		//TEST(w(array{ "a"s, map{ { "b"s, "c"s } } }) == "826161bf61626163ff");
		//TEST(w(map{ { "Fun"s, true },{ "Amt"s, -2ll } }) == "bf6346756ef563416d7421ff");
	}
}}