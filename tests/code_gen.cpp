#include "code_gen.h"
#include "stream.h"
#include "unit_test.h"
#include <sstream>

namespace gold_fish
{
	namespace details
	{
		std::string parse_type(const dom::document& d)
		{
			if (d.is<std::string>())
			{
				auto&& x = d.as<std::string>();
				if (x == "int")
					return "simple_type<tags::signed_int>";
				else if (x == "uint")
					return "simple_type<tags::unsigned_int>";
				else if (x == "float")
					return "simple_type<tags::floating_point>";
				else if (x == "bool")
					return "simple_type<tags::boolean>";
				else if (x == "blob")
					return "simple_type<tags::byte_string>";
				else if (x == "string")
					return "simple_type<tags::text_string>";
				else if (x == "document")
					return "untyped";
				else
					return "custom_type<" + x + ">";
			}
			else if (d.is<dom::array>())
			{
				if (d.as<dom::array>().size() != 1)
					throw bad_type_declaration{};
				return "typed_array_type<" + parse_type(d.as<dom::array>().front()) + ">";
			}
			else if (d.is<dom::map>())
			{
				if (d.as<dom::map>().size() != 1)
					throw bad_type_declaration{};
				auto&& x = d.as<dom::map>().front();

				return "typed_map_type<" + parse_type(x.first) + ", " + parse_type(x.second) + ">";
			}
			else
			{
				throw bad_type_declaration{};
			}
		}
		TEST_CASE(test_parse_type)
		{
			auto parse_type_from_json = [](auto s)
			{
				return parse_type(dom::load_in_memory(json::read(stream::ref(stream::read_string_literal(s)))));
			};

			TEST(parse_type_from_json("\"int\"") == "simple_type<tags::signed_int>");
			TEST(parse_type_from_json("\"uint\"") == "simple_type<tags::unsigned_int>");
			TEST(parse_type_from_json("\"float\"") == "simple_type<tags::floating_point>");
			TEST(parse_type_from_json("\"bool\"") == "simple_type<tags::boolean>");
			TEST(parse_type_from_json("\"blob\"") == "simple_type<tags::byte_string>");
			TEST(parse_type_from_json("\"string\"") == "simple_type<tags::text_string>");
			TEST(parse_type_from_json("\"document\"") == "untyped");
			TEST(parse_type_from_json("\"point\"") == "custom_type<point>");

			TEST(parse_type_from_json("[\"document\"]") == "typed_array_type<untyped>");
			TEST(parse_type_from_json("[\"int\"]") == "typed_array_type<simple_type<tags::signed_int>>");
			TEST(parse_type_from_json("[[\"int\"]]") == "typed_array_type<typed_array_type<simple_type<tags::signed_int>>>");

			TEST(parse_type_from_json("{\"document\":\"document\"}") == "typed_map_type<untyped, untyped>");
			TEST(parse_type_from_json("{\"blob\":[\"string\"]}") == "typed_map_type<simple_type<tags::byte_string>, typed_array_type<simple_type<tags::text_string>>>");

			// A map from a map<array<int>,int> to int
			TEST(parse_type_from_json(R"json({{["int"]:"int"}:"int"})json") == "typed_map_type<typed_map_type<typed_array_type<simple_type<tags::signed_int>>, simple_type<tags::signed_int>>, simple_type<tags::signed_int>>");
		}
	}

	template <class Stream>
	auto generate_code(Stream& s)
	{
		std::stringstream ss;
		generate_code(ss, s);
		return ss.str();
	}

	TEST_CASE(test_generate_code)
	{
		TEST(generate_code(stream::ref(stream::read_string_literal(
			R"json({
				"point":{
					"x":{"type":"uint"},
					"y":{"type":"uint"}
				}
			})json"))) == R"cpp(template <class Map> class point
{
public:
	point(Map map)
		: m_map(std::move(map))
	{}
	auto x()
	{
		uint8_t buffer[2];
		if (!m_map.try_move_to(field_names(), buffer, 0, 1))
			throw bad_type_declaration{};
		return simple_type<tags::unsigned_int>::cast(m_map.read_value());
	}
	auto y()
	{
		uint8_t buffer[2];
		if (!m_map.try_move_to(field_names(), buffer, 1, 2))
			throw bad_type_declaration{};
		return simple_type<tags::unsigned_int>::cast(m_map.read_value());
	}
private:
	static auto field_names()
	{
		static const array_ref<const char> result[] = {
			array_ref<const char>("x").without_end(1),
			array_ref<const char>("y").without_end(1),
		};
		return make_array_ref(result);
	}
	filtered_map<Map> m_map;
};
)cpp");
	}

	// Copied version of the generated code
	template <class Map> class point
	{
	public:
		point(Map map)
			: m_map(std::move(map))
		{}
		auto x()
		{
			uint8_t buffer[2];
			if (!m_map.try_move_to(field_names(), buffer, 0, 1))
				throw bad_type_declaration{};
			return simple_type<tags::unsigned_int>::cast(m_map.read_value());
		}
		auto y()
		{
			uint8_t buffer[2];
			if (!m_map.try_move_to(field_names(), buffer, 1, 2))
				throw bad_type_declaration{};
			return simple_type<tags::unsigned_int>::cast(m_map.read_value());
		}
	private:
		static auto field_names()
		{
			static const array_ref<const char> result[] = {
				array_ref<const char>("x").without_end(1),
				array_ref<const char>("y").without_end(1),
			};
			return make_array_ref(result);
		}
		filtered_map<Map> m_map;
	};
	TEST_CASE(test_point)
	{
		//auto pt = custom_type<point>::cast(json::read(stream::ref(stream::read_string_literal(R"json({"x":2, "y":3})json"))));
		//TEST(pt.x() == 2);
		//TEST(pt.y() == 3);
		//sax::skip(pt);
	}
}