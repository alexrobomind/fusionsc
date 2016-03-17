#pragma once

#include "tags.h"
#include "stream.h"
#include "optional.h"
#include "base64_stream.h"
#include "buffered_stream.h"
#include <type_traits>

namespace goldfish
{
	struct integer_overflow_while_casting : exception {};

	template <bool _does_json_conversions, class... types>
	class document_impl
	{
	public:
		using tag = tags::document;
		template <class tag> using type_with_tag_t = tags::type_with_tag_t<tag, types...>;
		enum { does_json_conversions = _does_json_conversions };

		template <class... Args> document_impl(Args&&... args)
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
		auto as_string() { return std::move(m_data).as<type_with_tag_t<tags::string>>(); }
		auto as_binary(std::true_type /*does_json_conversion*/) { return stream::decode_base64(as_string()); }
		auto as_binary(std::false_type /*does_json_conversion*/) { return std::move(m_data).as<type_with_tag_t<tags::binary>>(); }
		auto as_binary() { return as_binary(std::integral_constant<bool, does_json_conversions>()); }
		auto as_array() { return std::move(m_data).as<type_with_tag_t<tags::array>>(); }
		auto as_map() { return std::move(m_data).as<type_with_tag_t<tags::map>>(); }

		// Floating point can be converted from an int
		auto as_double()
		{
			return visit(first_match(
				[](auto&& x, tags::floating_point) -> double { return x; },
				[](auto&& x, tags::unsigned_int) -> double { return static_cast<double>(x); },
				[](auto&& x, tags::signed_int) -> double { return static_cast<double>(x); },
				[](auto&& x, tags::string) -> double
				{
					if (!does_json_conversions)
						throw bad_variant_access();

					auto s = stream::buffer<8>(stream::ref(x));
					return json::read_number(s).visit([](auto&& x) -> double { return static_cast<double>(x); });
				},
				[](auto&&, auto) -> double { throw bad_variant_access{}; }
			));
		}
		
		// Unsigned ints can be converted from signed ints
		uint64_t as_uint()
		{
			return visit(first_match(
				[](auto&& x, tags::unsigned_int) -> uint64_t { return x; },
				[](auto&& x, tags::signed_int) -> uint64_t { return cast_signed_to_unsigned(x); },
				[](auto&& x, tags::string) -> uint64_t
				{
					if (!does_json_conversions)
						throw bad_variant_access();

					auto s = stream::buffer<8>(stream::ref(x));
					return json::read_number(s).visit(best_match(
						[](uint64_t x) { return x; },
						[](int64_t x) { return cast_signed_to_unsigned(x); },
						[](double x) -> uint64_t { throw bad_variant_access{}; }));
				},
				[](auto&&, auto) -> uint64_t { throw bad_variant_access{}; }
			));
		}
		
		// Signed ints can be converted from unsigned ints
		int64_t as_int()
		{
			return visit(first_match(
				[](auto&& x, tags::signed_int) { return x; },
				[](auto&& x, tags::unsigned_int) { return cast_unsigned_to_signed(x); },
				[](auto&& x, tags::string) -> int64_t
				{
					if (!does_json_conversions)
						throw bad_variant_access();

					auto s = stream::buffer<8>(stream::ref(x));
					return json::read_number(s).visit(best_match(
						[](int64_t x) { return x; },
						[](uint64_t x) { return cast_unsigned_to_signed(x); },
						[](double x) -> int64_t { throw bad_variant_access{}; }
					));
				},
				[](auto&&, auto) -> int64_t { throw bad_variant_access{}; }
			));
		}
		auto as_bool() { return as<tags::boolean>(); }
		bool is_undefined() { return m_data.is<type_with_tag_t<tags::undefined>>(); }
		bool is_null() { return m_data.is<type_with_tag_t<tags::null>>(); }

		template <class tag> bool is_exactly() { return m_data.is<type_with_tag_t<tag>>(); }

		using invalid_state = typename variant<types...>::invalid_state;
	private:
		static uint64_t cast_signed_to_unsigned(int64_t x)
		{
			if (x < 0)
				throw integer_overflow_while_casting{};
			return static_cast<uint64_t>(x);
		}
		static int64_t cast_unsigned_to_signed(uint64_t x)
		{
			if (x > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
				throw integer_overflow_while_casting{};
			return static_cast<int64_t>(x);
		}

		variant<types...> m_data;
	};

	template <class Document> std::enable_if_t<tags::has_tag<std::decay_t<Document>, tags::document>::value, void> seek_to_end(Document&& d)
	{
		d.visit([&](auto&& x, auto) { seek_to_end(std::forward<decltype(x)>(x)); });
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::undefined>::value, void> seek_to_end(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::floating_point>::value, void> seek_to_end(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::unsigned_int>::value, void> seek_to_end(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::signed_int>::value, void> seek_to_end(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::boolean>::value, void> seek_to_end(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::null>::value, void> seek_to_end(type&&) {}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::binary>::value, void> seek_to_end(type&& x)
	{
		stream::seek(x, std::numeric_limits<uint64_t>::max());
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::string>::value, void> seek_to_end(type&& x)
	{
		stream::seek(x, std::numeric_limits<uint64_t>::max());
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::array>::value, void> seek_to_end(type&& x)
	{
		while (auto d = x.read())
			seek_to_end(*d);
	}
	template <class type> std::enable_if_t<tags::has_tag<std::decay_t<type>, tags::map>::value, void> seek_to_end(type&& x)
	{
		while (auto d = x.read_key())
		{
			seek_to_end(*d);
			seek_to_end(x.read_value());
		}
	}
}