#pragma once

#include <vector>
#include <string>
#include "tags.h"
#include "stream.h"

namespace goldfish { namespace dom
{
	struct document;

	using array = std::vector<document>;
	using map = std::vector<std::pair<document, document>>;

	using map_key = variant<bool, nullptr_t, uint64_t, int64_t, double, std::vector<uint8_t>, std::string>;
	using document_variant = variant<
		bool,
		nullptr_t,
		tags::undefined,
		uint64_t,
		int64_t,
		double,
		std::vector<uint8_t>,
		std::string,
		array,
		map>;

	struct document : document_variant
	{
		template <class... Args> document(Args&&... args)
			: document_variant(std::forward<Args>(args)...)
		{}
	};

	inline document string(const char* data, size_t size) { return std::string{ data, data + size }; }
	inline document string(const char* data) { return string(data, strlen(data)); }
	inline document string(const std::string& data) { return string(data.data(), data.size()); }

	inline document binary(const uint8_t* data, size_t size) { return std::vector<uint8_t>{ data, data + size }; }
	inline document binary(const std::vector<uint8_t>& data) { return binary(data.data(), data.size()); }

	inline uint64_t load_in_memory(uint64_t d) { return d; }
	inline int64_t load_in_memory(int64_t d) { return d; }
	inline bool load_in_memory(bool d) { return d; }
	inline nullptr_t load_in_memory(nullptr_t d) { return d; }
	inline double load_in_memory(double d) { return d; }

	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::binary>::value, std::vector<uint8_t>> load_in_memory(D&& d)
	{
		return stream::read_all(d);
	}
	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::string>::value, std::string> load_in_memory(D&& d)
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

	inline tags::undefined load_in_memory(tags::undefined) { return tags::undefined{}; }

	struct unexpected_end_of_structure {};

	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::document>::value, document> load_in_memory(D&& d)
	{
		return d.visit([&](auto&& x, auto tag) -> document { return load_in_memory(x); });
	}
}}