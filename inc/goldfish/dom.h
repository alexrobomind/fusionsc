#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "variant.h"
#include "array_ref.h"
#include <type_traits>
#include "tags.h"

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
		using tag = tags::document;
		template <class... Args>
		document(Args&&... args)
			: document_variant(std::forward<Args>(args)...)
		{}
	};

	inline document string(const char* data, size_t size) { return std::string{ data, data + size }; }
	inline document string(const char* data) { return string(data, strlen(data)); }
	inline document string(const std::string& data) { return string(data.data(), data.size()); }

	inline document binary(const uint8_t* data, size_t size) { return std::vector<uint8_t>{ data, data + size }; }
	inline document binary(const std::vector<uint8_t>& data) { return binary(data.data(), data.size()); }
}}