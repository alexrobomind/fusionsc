#include "json_reader.h"
#include "sax_reader.h"
#include "unit_test.h"

namespace gold_fish
{
	size_t index_of(const_buffer_ref value, array_ref<const array_ref<const char>> keys)
	{
		return std::distance(keys.begin(), std::find_if(keys.begin(), keys.end(),
			[&](auto&& key)
		{
			return key.size() == value.size() && std::equal(value.begin(), value.end(), key.begin());
		}));
	}

	template <class... String> optional<std::pair<size_t, uint64_t>> test_skip_to_key_helper(const std::vector<array_ref<const char>>& keys)
	{
		uint8_t buffer[1024];

		auto s = stream::read_string_literal(R"json({"1":1,"12":2,"123":3})json");
		auto map = json::read(stream::ref(s));
		auto& x = map.as<tags::map>();

		auto index = skip_to_key(x, buffer, make_array_ref(keys));
		if (index == nullopt)
			return nullopt;
				
		auto result = std::make_pair(*index, x.read_value().as<tags::unsigned_int>());
		skip(map);
		return result;
	}
	TEST_CASE(test_skip_to_key)
	{
		TEST(test_skip_to_key_helper({ make_array_ref("1").without_end(1) }) == std::make_pair(0, 1));
		TEST(test_skip_to_key_helper({ make_array_ref("12").without_end(1) }) == std::make_pair(0, 2));
		TEST(test_skip_to_key_helper({ make_array_ref("123").without_end(1) }) == std::make_pair(0, 3));
		TEST(test_skip_to_key_helper({ make_array_ref("123").without_end(1), make_array_ref("12").without_end(1) }) == std::make_pair(1, 2));
		TEST(test_skip_to_key_helper({ make_array_ref("a").without_end(1) }) == nullopt);
	}

	class text
	{
		static auto test()
		{
			static const array_ref<const char> result[] = {
				array_ref<const char>("a").without_end(1),
				array_ref<const char>("ab").without_end(1)
			};
			return make_array_ref(result);
		}
	};
}