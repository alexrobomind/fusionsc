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

		template <class tag> bool is() const noexcept
		{
			static_assert(tags::is_tag<tag>::value, "document::is must be called with a tag (see tags.h)");
			return m_data.visit([&](auto&& x)
			{
				return std::is_same<tag, decltype(tags::get_tag(x))>::value;
			});
		}

		template <class tag> decltype(auto) as() & { return as_impl(tags::tag_t<tag>{}, std::integral_constant<bool, does_json_conversions>{}); }
		template <class tag> decltype(auto) as() && { return std::move(*this).as_impl(tags::tag_t<tag>{}, std::integral_constant<bool, does_json_conversions>{}); }

		using invalid_state = typename variant<types...>::invalid_state;
	private:
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

		// Byte strings are converted from text strings (assuming base64 text)
		auto as_impl(tags::byte_string, std::true_type /*json_conversion*/) &&
		{
			return base64(std::move(*this).as<tags::text_string>());
		}

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
}