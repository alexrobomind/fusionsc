#pragma once

#include "dom.h"
#include <vector>
#include <string>
#include "tags.h"
#include "stream.h"

namespace gold_fish { namespace dom
{
	inline uint64_t load_in_memory(uint64_t d) { return d; }
	inline int64_t load_in_memory(int64_t d) { return d; }
	inline bool load_in_memory(bool d) { return d; }
	inline nullptr_t load_in_memory(nullptr_t d) { return d; }
	inline double load_in_memory(double d) { return d; }

	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::byte_string>::value, std::vector<uint8_t>> load_in_memory(D&& d)
	{
		return stream::read_all(d);
	}
	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::text_string>::value, std::string> load_in_memory(D&& d)
	{
		return stream::read_all_as_string(d);
	}
	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::array>::value, array> load_in_memory(D&& d)
	{
		array result;
		while (auto x = d.read())
			result.emplace_back(load_in_memory(*x));
		return result;
	}
	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::map>::value, map> load_in_memory(D&& d)
	{
		map result;
		while (auto x = d.read_key())
		{
			auto key = load_in_memory(*x);
			result.emplace_back(key, load_in_memory(d.read_value()));
		}
		return result;
	}

	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::undefined>::value, undefined> load_in_memory(D&& d) { return undefined{}; }

	struct unexpected_end_of_structure {};

	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::document>::value, document> load_in_memory(D&& d)
	{
		return d.visit([&](auto&& x, auto tag) -> document { return load_in_memory(x); });
	}
}}