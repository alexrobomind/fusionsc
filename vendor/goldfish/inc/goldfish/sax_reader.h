#pragma once

#include "tags.h"
#include "stream.h"
#include "optional.h"
#include "base64_stream.h"
#include "buffered_stream.h"
#include "schema.h"
#include <type_traits>

namespace goldfish
{
	struct integer_overflow_while_casting : exception { integer_overflow_while_casting() : exception("Integer too large") {} };

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
			assert(!m_moved_from);
			return m_data.visit([&](auto& x) -> decltype(auto)
			{
				return l(x, tags::get_tag(x));
			});
		}
		template <class Lambda> decltype(auto) visit(Lambda&& l) &&
		{
			assert(!m_moved_from);
			return std::move(m_data).visit([&](auto&& x) -> decltype(auto)
			{
				return l(std::forward<decltype(x)>(x), tags::get_tag(x));
			});
		}
		auto as_string()
		{
			assert(!m_moved_from);
			#ifndef NDEBUG
			m_moved_from = true;
			#endif
			return std::move(m_data).as<type_with_tag_t<tags::string>>();
		}
		auto as_binary() { return as_binary(std::integral_constant<bool, does_json_conversions>()); }
		auto as_array()
		{
			assert(!m_moved_from);
			#ifndef NDEBUG
			m_moved_from = true;
			#endif
			return std::move(m_data).as<type_with_tag_t<tags::array>>();
		}
		auto as_map()
		{
			assert(!m_moved_from);
			#ifndef NDEBUG
			m_moved_from = true;
			#endif
			return std::move(m_data).as<type_with_tag_t<tags::map>>();
		}
		template <class FirstKey, class... OtherKeys> auto as_map(FirstKey&& first_key, OtherKeys&&... other_keys)
		{
			return apply_schema(as_map(), make_schema(std::forward<FirstKey>(first_key), std::forward<OtherKeys>(other_keys)...));
		}
		template <class... Args> auto as_object(Args&&... args) { return as_map(std::forward<Args>(args)...); }

		// Floating point can be converted from an int
		auto as_double()
		{
			assert(!m_moved_from);
			auto result = visit(first_match(
				[](double x, tags::floating_point) { return x; },
				[](auto&& x, tags::unsigned_int) { return static_cast<double>(x); },
				[](auto&& x, tags::signed_int) { return static_cast<double>(x); },
				[](auto&& x, tags::string)
				{
					// We need to buffer the stream because read_number uses "peek<char>"
					auto s = stream::buffer<1>(stream::ref(x));
					try
					{
						auto result = json::read_number(s, stream::read<char>(s)).visit([](auto&& x) -> double { return static_cast<double>(x); });
						if (stream::seek(s, 1) != 0)
							throw bad_variant_access{};
						return result;
					}
					catch (const json::ill_formatted_json_data&)
					{
						throw bad_variant_access{};
					}
					catch (const stream::unexpected_end_of_stream&)
					{
						throw bad_variant_access{};
					}
				},
				[](auto&&, auto) -> double { throw bad_variant_access{}; }
			));
			#ifndef NDEBUG
			m_moved_from = true;
			#endif
			return result;
		}

		// Unsigned ints can be converted from signed ints
		uint64_t as_uint64()
		{
			assert(!m_moved_from);
			auto result = visit(first_match(
				[](auto&& x, tags::unsigned_int) { return x; },
				[](auto&& x, tags::signed_int) { return cast_signed_to_unsigned(x); },
				[](auto&& x, tags::floating_point) { return cast_double_to_unsigned(x); },
				[](auto&& x, tags::string)
				{
					// We need to buffer the stream because read_number uses "peek<char>"
					auto s = stream::buffer<1>(stream::ref(x));
					try
					{
						auto result = json::read_number(s, stream::read<char>(s)).visit(best_match(
							[](uint64_t x) { return x; },
							[](int64_t x) { return cast_signed_to_unsigned(x); },
							[](double x) { return cast_double_to_unsigned(x); }));
						if (stream::seek(s, 1) != 0)
							throw bad_variant_access{};
						return result;
					}
					catch (const json::ill_formatted_json_data&)
					{
						throw bad_variant_access{};
					}
					catch (const stream::unexpected_end_of_stream&)
					{
						throw bad_variant_access{};
					}
				},
				[](auto&&, auto) -> uint64_t { throw bad_variant_access{}; }
			));
			#ifndef NDEBUG
			m_moved_from = true;
			#endif
			return result;
		}
		uint32_t as_uint32()
		{
			auto x = as_uint64();
			if (x > std::numeric_limits<uint32_t>::max())
				throw integer_overflow_while_casting{};
			return static_cast<uint32_t>(x);
		}
		uint16_t as_uint16()
		{
			auto x = as_uint64();
			if (x > std::numeric_limits<uint16_t>::max())
				throw integer_overflow_while_casting{};
			return static_cast<uint16_t>(x);
		}
		uint8_t as_uint8()
		{
			auto x = as_uint64();
			if (x > std::numeric_limits<uint8_t>::max())
				throw integer_overflow_while_casting{};
			return static_cast<uint8_t>(x);
		}

		// Signed ints can be converted from unsigned ints
		int64_t as_int64()
		{
			assert(!m_moved_from);
			auto result = visit(first_match(
				[](auto&& x, tags::signed_int) { return x; },
				[](auto&& x, tags::unsigned_int) { return cast_unsigned_to_signed(x); },
				[](auto&& x, tags::floating_point) { return cast_double_to_signed(x); },
				[](auto&& x, tags::string)
				{
					// We need to buffer the stream because read_number uses "peek<char>"
					auto s = stream::buffer<1>(stream::ref(x));
					try
					{
						auto result = json::read_number(s, stream::read<char>(s)).visit(best_match(
							[](int64_t x) { return x; },
							[](uint64_t x) { return cast_unsigned_to_signed(x); },
							[](double x) { return cast_double_to_signed(x); }
						));
						if (stream::seek(s, 1) != 0)
							throw bad_variant_access{};
						return result;
					}
					catch (const json::ill_formatted_json_data&)
					{
						throw bad_variant_access{};
					}
					catch (const stream::unexpected_end_of_stream&)
					{
						throw bad_variant_access{};
					}
				},
				[](auto&&, auto) -> int64_t { throw bad_variant_access{}; }
			));
			#ifndef NDEBUG
			m_moved_from = true;
			#endif
			return result;
		}
		int32_t as_int32()
		{
			auto x = as_int64();
			if (x < std::numeric_limits<int32_t>::min() || x > std::numeric_limits<int32_t>::max())
				throw integer_overflow_while_casting{};
			return static_cast<int32_t>(x);
		}
		int16_t as_int16()
		{
			auto x = as_int64();
			if (x < std::numeric_limits<int16_t>::min() || x > std::numeric_limits<int16_t>::max())
				throw integer_overflow_while_casting{};
			return static_cast<int16_t>(x);
		}
		int8_t as_int8()
		{
			auto x = as_int64();
			if (x < std::numeric_limits<int8_t>::min() || x > std::numeric_limits<int8_t>::max())
				throw integer_overflow_while_casting{};
			return static_cast<int8_t>(x);
		}

		auto as_bool()
		{
			assert(!m_moved_from);
			auto result = visit(first_match(
				[](auto&& x, tags::boolean) { return x; },
				[](auto&& x, tags::string)
				{
					byte buffer[6];
					auto cb = read_full_buffer(x, buffer);
					if (cb == 4 && std::equal(buffer, buffer + 4, "true"))
						return true;
					else if (cb == 5 && std::equal(buffer, buffer + 5, "false"))
						return false;
					else
						throw bad_variant_access{};
				},
				[](auto&&, auto) -> bool { throw bad_variant_access{}; }
			));
			#ifndef NDEBUG
			m_moved_from = true;
			#endif
			return result;
		}
		bool is_undefined_or_null() const { return m_data.is<undefined>() || m_data.is<nullptr_t>(); }
		bool is_null() const { return m_data.is<nullptr_t>(); }

		template <class tag> bool is_exactly() { return m_data.is<type_with_tag_t<tag>>(); }

		#ifdef NDEBUG
		using invalid_state = typename variant<types...>::invalid_state;
		#endif
	private:
		auto as_binary(std::true_type /*does_json_conversion*/) { return stream::decode_base64(as_string()); }
		auto as_binary(std::false_type /*does_json_conversion*/) { return std::move(m_data).as<type_with_tag_t<tags::binary>>(); }

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
		static uint64_t cast_double_to_unsigned(double x)
		{
			if (x == static_cast<uint64_t>(x))
				return static_cast<uint64_t>(x);
			else
				throw integer_overflow_while_casting{};
		}
		static int64_t cast_double_to_signed(double x)
		{
			if (x == static_cast<int64_t>(x))
				return static_cast<int64_t>(x);
			else
				throw integer_overflow_while_casting{};
		}

		#ifndef NDEBUG
		bool m_moved_from = false;
		#endif
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