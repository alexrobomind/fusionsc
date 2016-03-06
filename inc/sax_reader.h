#pragma once

#include "tags.h"
#include "stream.h"
#include "optional.h"

namespace gold_fish
{
	template <class... types>
	class document_on_variant
	{
	public:
		using tag = tags::document;
		template <class... Args> document_on_variant(Args&&... args)
			: m_data(std::forward<Args>(args)...)
		{}

		template <class Lambda> decltype(auto) visit(Lambda&& l) &
		{
			return m_data.visit([&](auto& x) -> decltype(auto)
			{
				return l(x, tags::get_tag(x));
			});
		}
		template <class Lambda> decltype(auto) visit(Lambda&& l) &&
		{
			return std::move(m_data).visit([&](auto&& x) -> decltype(auto)
			{
				return l(std::forward<decltype(x)>(x), tags::get_tag(x));
			});
		}
		template <class tag> auto& as() noexcept
		{
			return m_data.as<tags::type_with_tag_t<tag, types...>>();
		}
		template <class tag> bool is() const noexcept
		{
			static_assert(tags::is_tag<tag>::value, "document::is must be called with a tag (see tags.h)");
			return m_data.visit([&](auto&& x)
			{
				return std::is_same<tag, decltype(tags::get_tag(x))>::value;
			});
		}
		template <class T> bool is_exactly() const noexcept
		{
			return m_data.is<T>();
		}

		using invalid_state = typename variant<types...>::invalid_state;
	private:
		variant<types...> m_data;
	};
	
	template <class Document> void skip(Document&& d);

	template <class type> void skip(type&& x, tags::undefined) {}
	template <class type> void skip(type&& x, tags::floating_point) {}
	template <class type> void skip(type&& x, tags::unsigned_int) {}
	template <class type> void skip(type&& x, tags::signed_int) {}
	template <class type> void skip(type&& x, tags::boolean) {}
	template <class type> void skip(type&& x, tags::null) {}
	template <class type> void skip(type&& x, tags::byte_string)
	{
		stream::skip(x, std::numeric_limits<uint64_t>::max());
	}
	template <class type> void skip(type&& x, tags::text_string)
	{
		stream::skip(x, std::numeric_limits<uint64_t>::max());
	}
	template <class type> void skip(type&& x, tags::array)
	{
		while (auto d = x.read())
			skip(*d);
	}

	template <class type> void skip(type&& x, tags::map)
	{
		while (auto d = x.read_key())
		{
			skip(*d);
			skip(x.read_value());
		}
	}

	template <class Document> void skip(Document&& d)
	{
		d.visit([&](auto&& x, auto tag) { skip(std::forward<decltype(x)>(x), tags::get_tag(x)); });
	}
	
	size_t index_of(const_buffer_ref value, array_ref<const array_ref<const char>> keys);

	template <class T> struct number_of_characters {};
	template <size_t N> struct number_of_characters<const char(&)[N]> { enum { value = N - 1 }; };
	template <class First> constexpr auto constexpr_max(First v) { return v; }
	template <class First, class Second> constexpr auto constexpr_max(First x, Second y) { return x < y ? y : x; }
	template <class First, class... Values> constexpr auto constexpr_max(First head, Values... v)
	{
		return constexpr_max(head, constexpr_max(v...));
	}
	template <class Map> std::enable_if_t<tags::has_tag<Map, tags::map>::value, optional<size_t>> skip_to_key(Map& map, buffer_ref buffer, array_ref<const array_ref<const char>> keys)
	{
		while (auto key = map.read_key())
		{
			auto index_match = key->visit(first_match(
				[&](auto&& value, tags::text_string)
				{
					auto cb = value.read_buffer(buffer);
					return index_of({ buffer.data(), cb }, keys);
				},
				[&](auto&&, auto&&) { return keys.size(); }
			));
			skip(*key);
			if (index_match != keys.size())
				return index_match;
			skip(map.read_value());
		}
		return nullopt;
	}

	struct untyped
	{
		template <class Document> static decltype(auto) cast(Document& d) { return d; }
	};
	template <class Tag> struct simple_type
	{
		template <class Document> static decltype(auto) cast(Document& d) { return d.as<Tag>(); }
	};
	template <template <class Map> class T> struct custom_type
	{
		template <class Document> static auto cast(Document& d) { return T<std::decay_t<decltype(d.as<tags::map>())>>{ d.as<tags::map>() }; }
	};

	template <class SubType, class Array> class typed_array
	{
	public:
		typed_array(Array array)
			: m_array(std::move(array))
		{}

		auto read() -> optional<std::decay_t<decltype(SubType::cast(std::declval<Array>().read()))>>
		{
			auto d = m_array.read();
			if (d)
				return SubType::cast(d);
			else
				return nullopt;
		}
	private:
		Array m_array;
	};
	template <class SubType, class Array> typed_array<SubType, std::decay_t<Array>> make_typed_array(Array&& array) { return{ std::forward<Array>(array) }; }
	template <class SubType> struct typed_array_type
	{
		template <class Document> static auto cast(Document& d) { return make_typed_array<SubType>(d.as<tags::array>()); }
	};

	template <class Map, class SubTypeKey, class SubTypeValue> class typed_map
	{
	public:
		typed_map(Map map)
			: m_map(std::move(map))
		{}

		auto read_key() -> optional<std::decay_t<decltype(SubTypeKey::cast(std::declval<Map>().read_key()))>>
		{
			auto d = m_array.read_key();
			if (d)
				return SubTypeKey::cast(d);
			else
				return nullopt;
		}
		auto read_value() { return SubTypeValue::cast(m_map.read_value()); }
	private:
		Map m_map;
	};
	template <class SubTypeKey, class SubTypeValue, class Map> typed_map<SubTypeKey, SubTypeValue, std::decay_t<Map>> make_typed_map(Map&& map) { return{ std::forward<Map>(map) }; }
	template <class SubTypeKey, class SubTypeValue> struct typed_map_type
	{
		template <class Document> static auto cast(Document& d) { return make_typed_map<SubTypeKey, SubTypeValue>(d.as<tags::map>()); }
	};

	template <class Map> class filtered_map
	{
	public:
		filtered_map(Map&& map)
			: m_map(std::move(map))
		{}
		bool try_move_to(array_ref<const array_ref<const char>> key_names, buffer_ref buffer, size_t desired_field_index, size_t max_field_index)
		{
			if (m_next_index > desired_field_index)
				return false;

			if (auto i = skip_to_key(m_map, buffer, key_names.slice(desired_field_index, max_field_index)))
			{
				m_next_index = desired_field_index + 1 + *i;
				return *i == 0;
			}
			else
			{
				m_next_index = key_names.size();
				return false;
			}
		}
		auto read_value() { return m_map.read_value(); }
	private:
		Map m_map;
		size_t m_next_index = 0;
	};

}