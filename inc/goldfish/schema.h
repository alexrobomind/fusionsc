#pragma once

#include "array_ref.h"
#include "optional.h"
#include "tags.h"
#include <vector>

namespace goldfish
{
	class schema
	{
		static const size_t max_length = 8 * 1024;
		
	public:
		schema(std::initializer_list<array_ref<const char>> key_names)
		{
			m_key_names.reserve(key_names.size());
			for (auto&& name : key_names)
			{
				assert(name.size() <= max_length);
				assert(name.back() == '\0');
				m_key_names.push_back(name.without_end(1));
			}
		}

		template <size_t N> optional<size_t> search_text(const char(&text)[N]) const
		{
			auto it = std::find_if(m_key_names.begin(), m_key_names.end(), [&](auto key_name)
			{
				return key_name.size() == N - 1 &&
					std::equal(key_name.begin(), key_name.end(), text);
			});
			if (it == m_key_names.end())
				return nullopt;
			else
				return std::distance(m_key_names.begin(), it);
		}
		template <class Document> std::enable_if_t<tags::has_tag<Document, tags::document>::value, optional<size_t>> search(Document& d) const
		{
			return d.visit(first_match(
				[&](auto& text, tags::string) -> optional<size_t>
				{
					uint8_t buffer[max_length];
					auto length = text.read_buffer(buffer);
					if (stream::seek(text, std::numeric_limits<uint64_t>::max()) != 0)
						return nullopt;

					auto it = std::find_if(m_key_names.begin(), m_key_names.end(), [&](auto key_name)
					{
						return key_name.size() == length &&
							std::equal(key_name.begin(), key_name.end(), buffer);
					});
					if (it == m_key_names.end())
						return nullopt;
					else
						return std::distance(m_key_names.begin(), it);
				},
				[&](auto&, auto) -> optional<size_t>
				{
					seek_to_end(d);
					return nullopt; /*We currently only support text strings as keys*/
				}));
		}
	private:
		std::vector<array_ref<const char>> m_key_names;
	};

	template <class Map> class map_with_schema
	{
	public:
		map_with_schema(Map&& map, const schema& s)
			: m_map(std::move(map))
			, m_schema(s)
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
				if (auto index = m_schema.search(*key))
				{
					m_index = *index;
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
		const schema& m_schema;
		size_t m_index = 0;
		bool m_on_value = false;
	};
	template <class Map> map_with_schema<std::decay_t<Map>> apply_schema(Map&& map, const schema& s)
	{
		return{ std::forward<Map>(map), s };
	}
}