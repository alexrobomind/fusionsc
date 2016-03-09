#pragma once

#include "tags.h"
#include "stream.h"
#include "optional.h"
#include "base64_stream.h"
#include "buffered_stream.h"
#include "typed_erased_stream.h"
#include <type_traits>

namespace goldfish
{
	struct integer_overflow {};

	template <bool _does_json_conversions, class... types>
	class document_on_variant
	{
	public:
		using tag = tags::document;
		template <class tag> using type_with_tag_t = tags::type_with_tag_t<tag, types...>;
		enum { does_json_conversions = _does_json_conversions };

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

		template <class tag> decltype(auto) as() & { return as_impl(tag{}, std::integral_constant<bool, does_json_conversions>{}); }
		template <class tag> decltype(auto) as() && { return std::move(*this).as_impl(tag{}, std::integral_constant<bool, does_json_conversions>{}); }

		// Default: no conversion
		template <class tag, class json_conversion> decltype(auto) as_impl(tag, json_conversion) &
		{
			return m_data.as<type_with_tag_t<tag>>();
		}
		template <class tag, class json_conversion> decltype(auto) as_impl(tag, json_conversion) &&
		{
			return std::move(m_data).as<type_with_tag_t<tag>>();
		}

		// Floating point can be converted from an int
		template <class json_conversion> double as_impl(tags::floating_point, json_conversion)
		{
			return visit(first_match(
				[](auto&& x, tags::floating_point) -> double { return x; },
				[](auto&& x, tags::unsigned_int) -> double { return static_cast<double>(x); },
				[](auto&& x, tags::signed_int) -> double { return static_cast<double>(x); },
				[](auto&& x, tags::text_string) -> double
				{
					if (!does_json_conversions)
						throw bad_variant_access();

					auto s = stream::buffer<8>(stream::ref(x));
					return json::read_number(s).visit([](auto&& x) -> double { return static_cast<double>(x); });
				},
				[](auto&&, auto) -> double { throw bad_variant_access{}; }
			));
		}
		
		// Signed ints can be converted from unsigned ints
		template <class json_conversion> int64_t as_impl(tags::signed_int, json_conversion)
		{
			return visit(first_match(
				[](auto&& x, tags::signed_int) -> int64_t { return x; },
				[](auto&& x, tags::unsigned_int) -> int64_t
				{
					if (x > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
						throw integer_overflow{};
					return static_cast<int64_t>(x);
				},
				[](auto&& x, tags::text_string) -> int64_t
				{
					if (!does_json_conversions)
						throw bad_variant_access();

					auto s = stream::buffer<8>(stream::ref(x));
					return json::read_number(s).visit([](auto&& x) -> int64_t { return static_cast<int64_t>(x); });
				},
				[](auto&&, auto) -> int64_t { throw bad_variant_access{}; }
			));
		}

		// Byte strings can be converted from text strings (assuming base64 text)
		stream::typed_erased_reader as_impl(tags::byte_string, std::true_type /*json_conversion*/) &&
		{
			return std::move(*this).visit(first_match(
				[](auto&& x, tags::byte_string) -> stream::typed_erased_reader { return stream::erase_type(std::forward<decltype(x)>(x)); },
				[](auto&& x, tags::text_string) -> stream::typed_erased_reader { return stream::erase_type(base64(std::forward<decltype(x)>(x))); },
				[](auto&&, auto) -> stream::typed_erased_reader { throw bad_variant_access{}; }
			));
		}
				
		template <class tag> bool is() const noexcept
		{
			static_assert(tags::is_tag<tag>::value, "document::is must be called with a tag (see tags.h)");
			return m_data.visit([&](auto&& x)
			{
				return std::is_same<tag, decltype(tags::get_tag(x))>::value;
			});
		}

		using invalid_state = typename variant<types...>::invalid_state;
	private:
		variant<types...> m_data;
	};

	template <class Document> std::enable_if_t<tags::has_tag<std::decay_t<Document>, tags::document>::value, void> skip(Document&& d)
	{
		d.visit([&](auto&& x, auto) { skip(std::forward<decltype(x)>(x)); });
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::undefined>::value, void> skip(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::floating_point>::value, void> skip(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::unsigned_int>::value, void> skip(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::signed_int>::value, void> skip(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::boolean>::value, void> skip(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::null>::value, void> skip(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::byte_string>::value, void> skip(type&& x)
	{
		stream::skip(x, std::numeric_limits<uint64_t>::max());
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::text_string>::value, void> skip(type&& x)
	{
		stream::skip(x, std::numeric_limits<uint64_t>::max());
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::array>::value, void> skip(type&& x)
	{
		while (auto d = x.read())
			skip(*d);
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::map>::value, void> skip(type&& x)
	{
		while (auto d = x.read_key())
		{
			skip(*d);
			skip(x.read_value());
		}
	}

	template <class Map> class filtered_map
	{
	public:
		filtered_map(Map&& map, array_ref<const uint64_t> key_names)
			: m_map(std::move(map))
			, m_key_names(key_names)
		{}
		optional<decltype(std::declval<Map>().read_value())> read_value_by_index(size_t index)
		{
			assert(m_index < m_key_names.size());
			if (m_index > index)
				return nullopt;

			if (m_on_value)
			{
				m_on_value = false;
				if (m_index == index)
					return m_map.read_value();
				else
					skip(m_map.read_value());
			}
			assert(!m_on_value);

			while (auto key = m_map.read_key())
			{
				// We currently only support unsigned int key types
				if (!key->is<tags::unsigned_int>())
				{
					skip(*key);
					skip(m_map.read_value());
					continue;
				}

				// do any of the keys match?
				auto it = std::find(m_key_names.begin() + m_index, m_key_names.end(), key->as<tags::unsigned_int>());
				if (it == m_key_names.end())
				{
					// This was a new key that we didn't know about, skip it
					skip(m_map.read_value());
					continue;
				}

				// We found the key, compute its index
				m_index = std::distance(m_key_names.begin(), it);
				if (m_index == index)
				{
					// That's the key we were looking for, return its value
					// at that point, we assume not being on value any more because the caller will process the value
					assert(!m_on_value);
					return m_map.read_value();
				}
				else if (m_index > index)
				{
					// Our key was not found (we found a key later in the list of keys)
					// We are on the value of that later key
					m_on_value = true;
					return nullopt;
				}
				else
				{
					// We found a key that is still before us, skip the value and keep searching
					skip(m_map.read_value());
				}
			}

			return nullopt;
		}
		friend void skip(filtered_map& m)
		{
			if (m.m_on_value)
			{
				skip(m.m_map.read_value());
				m.m_on_value = false;
			}

			goldfish::skip(m.m_map);
			m.m_index = m.m_key_names.size();
		}
	private:
		Map m_map;
		array_ref<const uint64_t> m_key_names;
		size_t m_index = 0;
		bool m_on_value = false;
	};
	template <class Map> filtered_map<std::decay_t<Map>> filter_map(Map&& map, array_ref<const uint64_t> key_names)
	{
		return{ std::forward<Map>(map), key_names };
	}
}