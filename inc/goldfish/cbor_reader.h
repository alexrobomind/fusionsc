#pragma once

#include "debug_checks_reader.h"
#include "variant.h"
#include "optional.h"
#include "numbers.h"
#include "sax_reader.h"
#include "tags.h"
#include "match.h"
#include "stream.h"

namespace goldfish { namespace cbor
{
	struct ill_formatted {};

	template <class Stream, uint8_t expected_type, class _tag> class string;
	template <class Stream> using byte_string = string<Stream, 2, tags::byte_string>;
	template <class Stream> using text_string = string<Stream, 3, tags::text_string>;
	template <class Stream> class array;
	template <class Stream> class map;
	struct undefined { using tag = tags::undefined; };

	template <class Stream> using document = document_on_variant<
		false /*does_json_conversions*/,
		bool,
		nullptr_t,
		uint64_t,
		int64_t,
		double,
		byte_string<Stream>,
		text_string<Stream>,
		array<Stream>,
		map<Stream>,
		undefined>;
	template <class Stream> optional<document<std::decay_t<Stream>>> read_no_debug_check(Stream&& s);

	template <class Stream, uint8_t expected_type, class _tag> class string
	{
	public:
		using tag = _tag;
		string(Stream&& s, bool single_block, uint64_t cb_initial)
			: m_stream(std::move(s))
			, m_single_block(single_block)
			, m_remaining_in_current_block(cb_initial)
		{}
		size_t read_buffer(buffer_ref buffer)
		{
			size_t cb_read = 0;
			while (!buffer.empty() && ensure_block())
			{
				auto to_read = static_cast<size_t>(std::min<uint64_t>(buffer.size(), m_remaining_in_current_block));
				if (m_stream.read_buffer(buffer.remove_front(to_read)) != to_read)
					throw ill_formatted{};
				m_remaining_in_current_block -= to_read;
				cb_read += to_read;
			}
			return cb_read;
		}

		uint64_t seek(uint64_t cb)
		{
			uint64_t original = cb;
			while (cb > 0 && ensure_block())
			{
				auto to_skip = std::min(cb, m_remaining_in_current_block);
				if (stream::seek(m_stream, to_skip) != to_skip)
					throw ill_formatted{};
				cb -= to_skip;
				m_remaining_in_current_block -= to_skip;
			}
			return original - cb;
		}

	private:
		bool ensure_block()
		{
			if (m_single_block)
			{
				return m_remaining_in_current_block > 0;
			}
			else
			{
				while (m_remaining_in_current_block == 0)
				{
					auto b = stream::read<uint8_t>(m_stream);
					if (b == 0xFF)
					{
						m_remaining_in_current_block = invalid_remaining;
						return false;
					}

					if ((b >> 5) != expected_type)
						throw ill_formatted{};

					m_remaining_in_current_block = read_integer(b & 31, m_stream);
					if (m_remaining_in_current_block == invalid_remaining)
						throw ill_formatted{};
				}
				return m_remaining_in_current_block != invalid_remaining;
			}
		}

		static const uint64_t invalid_remaining = 0x7FFFFFFFFFFFFFFFull;
		Stream m_stream;

		uint64_t m_single_block : 1;
		uint64_t m_remaining_in_current_block : 63;
	};

	template <class Stream> class array
	{
	public:
		array(const array&) = delete;
		array(array&&) = default;
		array(Stream&& s, uint64_t length)
			: m_stream(std::move(s))
			, m_remaining_length(length)
		{}
		using tag = tags::array;

		optional<document<stream::reader_ref_type_t<Stream>>> read()
		{
			if (m_remaining_length == 0)
				return nullopt;

			auto d = read_no_debug_check(stream::ref(m_stream));
			if (m_remaining_length != std::numeric_limits<uint64_t>::max())
				--m_remaining_length;
			if (!d)
			{
				if (m_remaining_length == std::numeric_limits<uint64_t>::max())
					m_remaining_length = 0;
				else
					throw ill_formatted{};
			}
			return d;
		}
	private:
		Stream m_stream;
		uint64_t m_remaining_length = std::numeric_limits<uint64_t>::max();
	};

	template <class Stream> class map
	{
	public:
		map(const map&) = delete;
		map(map&&) = default;
		map(Stream&& s, uint64_t remaining_length = std::numeric_limits<uint64_t>::max())
			: m_stream(std::move(s))
			, m_remaining_length(remaining_length)
		{}
		using tag = tags::map;

		optional<document<stream::reader_ref_type_t<Stream>>> read_key()
		{
			if (m_remaining_length == 0)
				return nullopt;

			auto d = read_no_debug_check(stream::ref(m_stream));
			if (!d)
			{
				if (m_remaining_length == std::numeric_limits<uint64_t>::max())
					m_remaining_length = 0;
				else
					throw ill_formatted{};
			}
			if (m_remaining_length != std::numeric_limits<uint64_t>::max())
				--m_remaining_length;
			return d;
		}
		document<stream::reader_ref_type_t<Stream>> read_value()
		{
			auto d = read_no_debug_check(stream::ref(m_stream));
			if (!d)
				throw ill_formatted{};
			return std::move(*d);
		}
	private:
		Stream m_stream;
		uint64_t m_remaining_length;
	};

	inline float to_float(uint32_t x) { return *reinterpret_cast<float*>(&x); }
	inline double to_double(uint64_t x) { return *reinterpret_cast<double*>(&x); }

	template <class Stream>
	uint64_t read_integer(uint8_t additional, Stream& s)
	{
		if (additional <= 23)
			return additional;
		else if (additional == 24)
			return read<uint8_t>(s);
		else if (additional == 25)
			return byte_swap(read<uint16_t>(s));
		else if (additional == 26)
			return byte_swap(read<uint32_t>(s));
		else if (additional == 27)
			return byte_swap(read<uint64_t>(s));
		else
			throw ill_formatted{};
	}
	template <class stream> double read_half_point_float(stream& s)
	{
		int half = (read<uint8_t>(s) << 8);
		half += read<uint8_t>(s);
		int exp = (half >> 10) & 0x1f;
		int mant = half & 0x3ff;
		double val;
		if (exp == 0) val = ldexp(mant, -24);
		else if (exp != 31) val = ldexp(mant + 1024, exp - 25);
		else val = mant == 0 ? INFINITY : NAN;
		return half & 0x8000 ? -val : val;
	}

	template <class Stream> struct read_helper
	{
		template <uint64_t value> static optional<document<Stream>> fn_uint(Stream&&, uint8_t) { return value; }
		static optional<document<Stream>> fn_small_int(Stream&& s, uint8_t b) { return static_cast<uint64_t>(b & 31); };
		static optional<document<Stream>> fn_int_8(Stream&& s, uint8_t b) { return static_cast<uint64_t>(stream::read<uint8_t>(s)); }
		static optional<document<Stream>> fn_int_16(Stream&& s, uint8_t b) { return static_cast<uint64_t>(byte_swap(stream::read<uint16_t>(s))); }
		static optional<document<Stream>> fn_int_32(Stream&& s, uint8_t b) { return static_cast<uint64_t>(byte_swap(stream::read<uint32_t>(s))); }
		static optional<document<Stream>> fn_int_64(Stream&& s, uint8_t b) { return byte_swap(stream::read<uint64_t>(s)); }
		static optional<document<Stream>> fn_neg_int(Stream&& s, uint8_t b)
		{
			auto x = read_integer(static_cast<uint8_t>(b & 31), s);
			if (x > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
				throw ill_formatted{};

			return -1 - static_cast<int64_t>(x);
		}

		static optional<document<Stream>> fn_tag(Stream&& s, uint8_t b)
		{
			// tags are skipped for now
			do
			{
				read_integer(static_cast<uint8_t>(b & 31), s);
				b = stream::read<uint8_t>(s);
			} while ((b >> 5) == 6);

			return read(std::forward<Stream>(s), b);
		}
		static optional<document<Stream>> fn_false(Stream&&, uint8_t) { return false; }
		static optional<document<Stream>> fn_true(Stream&&, uint8_t) { return true; }
		static optional<document<Stream>> fn_null(Stream&&, uint8_t) { return nullptr; }
		static optional<document<Stream>> fn_undefined(Stream&&, uint8_t) { return undefined{}; }
		static optional<document<Stream>> fn_end_of_structure(Stream&&, uint8_t) { return nullopt; }

		static optional<document<Stream>> fn_float_16(Stream&& s, uint8_t) { return read_half_point_float(s); }
		static optional<document<Stream>> fn_float_32(Stream&& s, uint8_t) { return double{ to_float(byte_swap(stream::read<uint32_t>(s))) }; }
		static optional<document<Stream>> fn_float_64(Stream&& s, uint8_t) { return to_double(byte_swap(stream::read<uint64_t>(s))); }
		static optional<document<Stream>> fn_ill_formatted(Stream&&, uint8_t) { throw ill_formatted{}; };

		static optional<document<Stream>> fn_small_byte(Stream&& s, uint8_t b) { return byte_string<Stream>{ std::move(s), true /*single_block*/, static_cast<uint8_t>(b & 31) }; };
		static optional<document<Stream>> fn_8_byte(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), true /*single_block*/, stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_byte(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), true /*single_block*/, byte_swap(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_byte(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), true /*single_block*/, byte_swap(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_byte(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), true /*single_block*/, byte_swap(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_undef_byte(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), false /*single_block*/, 0 }; };
		
		static optional<document<Stream>> fn_small_text(Stream&& s, uint8_t b) { return text_string<Stream>{ std::move(s), true /*single_block*/, static_cast<uint8_t>(b & 31) }; };
		static optional<document<Stream>> fn_8_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), true /*single_block*/, stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), true /*single_block*/, byte_swap(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), true /*single_block*/, byte_swap(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), true /*single_block*/, byte_swap(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_undef_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), false /*single_block*/, 0 }; };

		static optional<document<Stream>> fn_small_array(Stream&& s, uint8_t b) { return array<Stream>{ std::move(s), static_cast<uint8_t>(b & 31) }; };
		static optional<document<Stream>> fn_8_array(Stream&& s, uint8_t b) { return array<Stream>{ std::move(s), stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_array(Stream&& s, uint8_t b) { return array<Stream>{ std::move(s), byte_swap(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_array(Stream&& s, uint8_t b) { return array<Stream>{ std::move(s), byte_swap(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_array(Stream&& s, uint8_t b) { return array<Stream>{ std::move(s), byte_swap(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_undef_array(Stream&& s, uint8_t b) { return array<Stream>{ std::move(s), std::numeric_limits<uint64_t>::max() }; };

		static optional<document<Stream>> fn_small_map(Stream&& s, uint8_t b) { return map<Stream>{ std::move(s), static_cast<uint8_t>(b & 31) }; };
		static optional<document<Stream>> fn_8_map(Stream&& s, uint8_t b) { return map<Stream>{ std::move(s), stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_map(Stream&& s, uint8_t b) { return map<Stream>{ std::move(s), byte_swap(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_map(Stream&& s, uint8_t b) { return map<Stream>{ std::move(s), byte_swap(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_map(Stream&& s, uint8_t b) { return map<Stream>{ std::move(s), byte_swap(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_undef_map(Stream&& s, uint8_t b) { return map<Stream>{ std::move(s), std::numeric_limits<uint64_t>::max() }; };

		static optional<document<Stream>> read(Stream&& s, uint8_t b)
		{
			using fn = optional<document<Stream>> (*) (Stream&&, uint8_t);

			static fn functions[] = {
				// MAJOR TYPE 0
				fn_uint<0>  ,fn_uint<1>  ,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,
				fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,fn_small_int,
				fn_small_int,fn_small_int,fn_small_int,fn_small_int,
				fn_int_8,fn_int_16,fn_int_32,fn_int_64,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_ill_formatted,

				// MAJOR TYPE 1
				fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,
				fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,
				fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,
				fn_neg_int,fn_neg_int,

				// MAJOR TYPE 2
				fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,
				fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,
				fn_small_byte,fn_small_byte,fn_small_byte,fn_small_byte,
				fn_8_byte,fn_16_byte,fn_32_byte,fn_64_byte,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_undef_byte,

				// MAJOR TYPE 3
				fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,
				fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,
				fn_small_text,fn_small_text,fn_small_text,fn_small_text,
				fn_8_text,fn_16_text,fn_32_text,fn_64_text,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_undef_text,

				// MAJOR TYPE 4
				fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,
				fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,
				fn_small_array,fn_small_array,fn_small_array,fn_small_array,
				fn_8_array,fn_16_array,fn_32_array,fn_64_array,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_undef_array,

				// MAJOR TYPE 5
				fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,
				fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,
				fn_small_map,fn_small_map,fn_small_map,fn_small_map,
				fn_8_map,fn_16_map,fn_32_map,fn_64_map,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_undef_map,

				// MAJOR TYPE 6
				fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,
				fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,
				fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,fn_tag,
				fn_tag,fn_tag,

				// MAJOR TYPE 7
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_false, fn_true, fn_null, fn_undefined, fn_ill_formatted /*simple*/, fn_float_16, fn_float_32, fn_float_64,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_end_of_structure
			};
			static_assert(sizeof(functions) / sizeof(functions[0]) == 256, "The jump table should have 256 entries");
			return functions[b](std::move(s), b);
		}
	};
	template <class Stream> optional<document<std::decay_t<Stream>>> read_no_debug_check(Stream&& s)
	{
		return read_helper<std::decay_t<Stream>>::read(std::forward<Stream>(s), read<uint8_t>(s));
	}
	template <class Stream> auto read(Stream&& s)
	{
		auto d = read_no_debug_check(std::forward<Stream>(s));
		if (!d)
			throw ill_formatted{};
		return debug_check::add_read_checks(std::move(*d));
	}
}}