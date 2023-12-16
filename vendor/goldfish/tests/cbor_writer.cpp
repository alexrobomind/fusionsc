#include "dom.h"
#include <goldfish/cbor_writer.h>
#include <goldfish/stream.h>
#include "unit_test.h"

namespace goldfish { namespace dom
{
	static std::string to_hex_string(const std::vector<byte>& data)
	{
		std::string result;
		for (auto&& x : data)
		{
			result += "0123456789abcdef"[x >> 4];
			result += "0123456789abcdef"[x & 0b1111];
		}
		return result;
	}
	static uint8_t to_hex(char c)
	{
		if ('0' <= c && c <= '9') return c - '0';
		else if ('a' <= c && c <= 'f') return c - 'a' + 10;
		else if ('A' <= c && c <= 'F') return c - 'A' + 10;
		else std::terminate();
	};
	static auto to_vector(const std::string& input)
	{
		std::vector<byte> data;
		for (auto it = input.begin(); it != input.end(); it += 2)
		{
			uint8_t high = to_hex(*it);
			uint8_t low = to_hex(*next(it));
			data.push_back((high << 4) | low);
		}
		return data;
	};
	TEST_CASE(write_valid_examples)
	{
		auto w = [&](const document& d)
		{
			return to_hex_string(cbor::create_writer(stream::vector_writer{}).write(d));
		};
		test(w(0ull) == "00");
		test(w(1ull) == "01");
		test(w(10ull) == "0a");
		test(w(23ull) == "17");
		test(w(24ull) == "1818");
		test(w(25ull) == "1819");
		test(w(100ull) == "1864");
		test(w(1000ull) == "1903e8");
		test(w(1000000ull) == "1a000f4240");
		test(w(1000000000000ull) == "1b000000e8d4a51000");
		test(w(18446744073709551615ull) == "1bffffffffffffffff");
		test(w(-9223372036854775808ll) == "3b7fffffffffffffff");

		test(w(-1ll) == "20");
		test(w(1ll) == "01");
		test(w(-10ll) == "29");
		test(w(-100ll) == "3863");
		test(w(-1000ll) == "3903e7");
		test(w(-1000000ll) == "3a000f423f");

		test(w(1.1) == "fb3ff199999999999a");
		test(w(100000.0) == "fa47c35000");
		test(w(3.4028234663852886e+38) == "fa7f7fffff");
		test(w(1.0e+300) == "fb7e37e43c8800759c");
		test(w(-4.1) == "fbc010666666666666");
		test(w(std::numeric_limits<double>::quiet_NaN()) == "fb7ff8000000000000");
		test(w(std::numeric_limits<double>::infinity()) == "fa7f800000");
		test(w(-std::numeric_limits<double>::infinity()) == "faff800000");

		test(w(false) == "f4");
		test(w(true) == "f5");
		test(w(nullptr) == "f6");
		test(w(undefined{}) == "f7");
		
		test(w(to_vector("")) == "40");
		test(w(to_vector("01020304")) == "4401020304");

		test(w("") == "60");
		test(w("a") == "6161");
		test(w("IETF") == "6449455446");
		test(w("\"\\") == "62225c");
		test(w(u8"\u00fc") == "62c3bc");
		test(w(u8"\u6c34") == "63e6b0b4");

		test(w(array{}) == "80");
		test(w(array{ 1ull, 2ull, 3ull }) == "83010203");
		test(w(array{ 1ull, array{ 2ull, 3ull }, array{ 4ull, 5ull } }) == "8301820203820405");
		test(w(array{
			1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull,
				10ull, 11ull, 12ull, 13ull, 14ull, 15ull, 16ull,
				17ull, 18ull, 19ull, 20ull, 21ull, 22ull, 23ull,
				24ull, 25ull }) == "98190102030405060708090a0b0c0d0e0f101112131415161718181819");

		test(w(map{}) == "a0");
		test(w(map{ { 1ull, 2ull },{ 3ull, 4ull } }) == "a201020304");
		test(w(map{
			{ "a", 1ull },
			{ "b", array{ 2ull, 3ull } }
		}) == "a26161016162820203");
		test(w(array{ "a", map{ { "b", "c" } } }) == "826161a161626163");
		test(w(map{
			{ "a", "A" },
			{ "b", "B" },
			{ "c", "C" },
			{ "d", "D" },
			{ "e", "E" }
		}) == "a56161614161626142616361436164614461656145");
	}

	TEST_CASE(write_infinite_array)
	{
		auto w = [&](const std::vector<document>& data)
		{
			stream::vector_writer s;
			auto array = cbor::create_writer(stream::ref(s)).start_array();
			for (auto d : data)
				array.write(d);
			array.flush();
			return to_hex_string(s.flush());
		};
		test(w({}) == "9fff");

		test(w({
			1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull,
			9ull, 10ull, 11ull, 12ull, 13ull, 14ull, 15ull,
			16ull, 17ull, 18ull, 19ull, 20ull, 21ull, 22ull,
			23ull, 24ull, 25ull }) == "9f0102030405060708090a0b0c0d0e0f101112131415161718181819ff");
	}

	TEST_CASE(write_infinite_map)
	{
		auto w = [&](const std::vector<std::pair<document, document>>& data)
		{
			stream::vector_writer s;
			auto map = cbor::create_writer(stream::ref(s)).start_map();
			for (auto&& d : data)
				map.write(d.first, d.second);

			map.flush();
			return to_hex_string(s.flush());
		};

		test(w({
			{"Fun", true},
			{"Amt", -2ll}
		}) == "bf6346756ef563416d7421ff");
	}

	TEST_CASE(write_infinite_string)
	{
		auto w = [&](const std::vector<std::string>& data)
		{
			stream::vector_writer s;
			auto string = cbor::create_writer(stream::ref(s)).start_string();
			for (auto s : data)
				string.write_buffer({ reinterpret_cast<const byte*>(s.data()), s.size() });
			string.flush();
			return to_hex_string(s.flush());
		};

		test(w({ "strea", "ming" }) == "7f657374726561646d696e67ff");
	}

	TEST_CASE(write_infinite_binary_string)
	{
		auto w = [&](const std::vector<std::vector<byte>>& data)
		{
			stream::vector_writer s;
			auto binary = cbor::create_writer(stream::ref(s)).start_binary();
			for (auto d : data)
				binary.write_buffer(d);
			binary.flush();
			return to_hex_string(s.flush());
		};

		test(w({ to_vector("0102"), to_vector("030405") }) == "5f42010243030405ff");
	}
}}