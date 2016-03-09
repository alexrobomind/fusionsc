#pragma once

#include "array_ref.h"
#include "optional.h"

namespace goldfish
{
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