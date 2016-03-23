#pragma once

#include "array_ref.h"
#include "optional.h"
#include "tags.h"
#include <vector>

namespace goldfish
{
	namespace details
	{
		template <size_t N> const_buffer_ref make_key(const char(&text)[N])
		{
			static_assert(N > 0, "expect null terminated strings");
			assert(text[N - 1] == 0);
			return{ reinterpret_cast<const byte*>(text), N - 1 };
		}
		template <size_t N, size_t max_length> class schema
		{
		public:
			template <class... Args> schema(Args&&... args)
				: m_keys{ std::forward<Args>(args)... }
			{}
		
			template <size_t N> optional<size_t> search_text(const char(&text)[N]) const
			{
				return search_impl({ reinterpret_cast<const byte*>(text), N - 1 }, std::integral_constant<size_t, 0>{});
			}
			template <class Document> std::enable_if_t<tags::has_tag<Document, tags::document>::value, optional<size_t>> search(Document& d) const
			{
				return d.visit(first_match(
					[&](auto& text, tags::string) -> optional<size_t>
					{
						byte buffer[max_length];
						auto length = text.read_buffer(buffer);
						if (stream::seek(text, std::numeric_limits<uint64_t>::max()) != 0)
							return nullopt;

						return search_impl({ buffer, length }, std::integral_constant<size_t, 0>{});
					},
					[&](auto&, auto) -> optional<size_t>
					{
						seek_to_end(d);
						return nullopt; /*We currently only support text strings as keys*/
					}));
			}
		private:
			template <size_t cur_index> optional<size_t> search_impl(const_buffer_ref text, std::integral_constant<size_t, cur_index>) const
			{
				auto it = std::find_if(m_keys.begin(), m_keys.end(), [&](auto&& key)
				{
					return key.size() == text.size() && std::equal(key.begin(), key.end(), make_unchecked_array_iterator(text.begin()));
				});
				if (it == m_keys.end())
					return nullopt;
				else
					return std::distance(m_keys.begin(), it);
			}

			std::array<const_buffer_ref, N> m_keys;
		};
	}
	template <class... T> auto make_schema(T&&... keys)
	{
		return details::schema<sizeof...(T), 1024>(details::make_key(std::forward<T>(keys))...);
	}

	template <class Map, class Schema> class map_with_schema
	{
	public:
		map_with_schema(Map&& map, const Schema& schema)
			: m_map(std::move(map))
			, m_schema(schema)
		{}
		optional<decltype(std::declval<Map>().read_value())> read_value_by_index(size_t index)
		{
			if (m_index > index)
				return nullopt;

			if (m_on_value)
			{
				m_on_value = false;
				if (m_index == index)
					return m_map.read_value();
				else
					seek_to_end(m_map.read_value());
			}
			assert(!m_on_value);

			while (auto key = m_map.read_key())
			{
				if (auto new_index = m_schema.search(*key))
				{
					m_index = *new_index;
				}
				else
				{
					seek_to_end(m_map.read_value());
					continue;
				}

				// We found a key in the schema, is it the right one?
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
					seek_to_end(m_map.read_value());
				}
			}

			return nullopt;
		}
		template <size_t N> auto read_value(const char(&text)[N])
		{
			if (auto index = m_schema.search_text(text))
				return read_value_by_index(*index);
			else
				std::terminate();			
		}
		friend void seek_to_end(map_with_schema& m)
		{
			if (m.m_on_value)
			{
				goldfish::seek_to_end(m.m_map.read_value());
				m.m_on_value = false;
			}

			goldfish::seek_to_end(m.m_map);
		}
	private:
		Map m_map;
		Schema m_schema;
		size_t m_index = 0;
		bool m_on_value = false;
	};
	template <class Map, class Schema> map_with_schema<std::decay_t<Map>, Schema> apply_schema(Map&& map, const Schema& s)
	{
		return{ std::forward<Map>(map), s };
	}
}