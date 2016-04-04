#pragma once

#include <vector>
#include <string>
#include <goldfish/match.h>
#include <goldfish/tags.h>
#include <goldfish/stream.h>

namespace goldfish { namespace dom
{
	struct document;

	using array = std::vector<document>;
	using map = std::vector<std::pair<document, document>>;

	using map_key = variant<bool, nullptr_t, uint64_t, int64_t, double, std::vector<byte>, std::string>;
	using document_variant = variant<
		bool,
		nullptr_t,
		undefined,
		uint64_t,
		int64_t,
		double,
		std::vector<byte>,
		std::string,
		array,
		map>;

	// This struct is necessary in order to be able to forward declare "document",
	// which is necessary in order to define array
	struct document : private document_variant
	{
		document(const char* s)
			: document(std::string(s))
		{}
		template <class... Args> document(Args&&... args)
			: document_variant(std::forward<Args>(args)...)
		{}
		using document_variant::as;
		using document_variant::is;
		using document_variant::visit;


		// variant<bool, std::string>::operator== is dangerous: x == "" resolves as
		// x == (bool)"". By forcing the conversion to document, we avoid this problem
		friend bool operator == (const document& lhs, const document& rhs)
		{
			return static_cast<const document_variant&>(lhs) == static_cast<const document_variant&>(rhs);
		}
	};

	template <class Writer> auto serialize_to_goldfish(Writer& writer, const dom::document& document)
	{
		return document.visit(best_match(
			[&](bool x) { return writer.write(x); },
			[&](nullptr_t x) { return writer.write(x); },
			[&](undefined x) { return writer.write(x); },
			[&](uint64_t x) { return writer.write(x); },
			[&](int64_t x) { return writer.write(x); },
			[&](double x) { return writer.write(x); },
			[&](const std::vector<byte>& x) { return writer.write(const_buffer_ref{ x }); },
			[&](const std::string& x) { return writer.write(x); },
			[&](const dom::array& x)
			{
				auto array_writer = writer.start_array(x.size());
				for (auto&& y : x)
					array_writer.write(y);
				return array_writer.flush();
			},
			[&](const dom::map& x)
			{
				auto map_writer = writer.start_map(x.size());
				for (auto&& y : x)
					map_writer.write(y.first, y.second);
				return map_writer.flush();
			}));
	}

	template <class D> std::enable_if_t<tags::has_tag<std::decay_t<D>, tags::document>::value, document> load_in_memory(D&& reader)
	{
		return std::forward<D>(reader).visit(first_match(
			[](auto&& d, tags::binary) -> document { return stream::read_all(d); },
			[](auto&& d, tags::string) -> document { return stream::read_all_as_string(d); },
			[](auto&& d, tags::array) -> document
			{
				array result;
				while (auto x = d.read())
					result.emplace_back(load_in_memory(*x));
				return result;
			},
			[](auto&& d, tags::map) -> document
			{
				map result;
				while (auto x = d.read_key())
				{
					auto key = load_in_memory(*x);
					result.emplace_back(key, load_in_memory(d.read_value()));
				}
				return result;
			},
			[](auto&& x, auto) -> document { return std::forward<decltype(x)>(x); }
		));
	}
}}