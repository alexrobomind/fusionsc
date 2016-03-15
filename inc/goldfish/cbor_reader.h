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
	template <class Stream> using byte_string = string<Stream, 2, tags::binary>;
	template <class Stream> using text_string = string<Stream, 3, tags::string>;
	template <class Stream> class array;
	template <class Stream> class map;

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
		tags::undefined>;

	// Read one document from the stream, or return nullopt for the null terminator (byte 0xFF)
	template <class Stream> optional<document<std::decay_t<Stream>>> read_no_debug_check(Stream&& s);

	template <class Stream, uint8_t expected_type, class _tag> class string
	{
	public:
		using tag = _tag;
		string(Stream&& s)
			: m_stream(std::move(s))
			, m_single_block(false)
			, m_remaining_in_current_block(0)
		{}
		string(Stream&& s, uint64_t cb_initial)
			: m_stream(std::move(s))
			, m_single_block(true)
			, m_remaining_in_current_block(cb_initial)
		{
			if (m_remaining_in_current_block >= invalid_remaining)
				throw ill_formatted{};
		}

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

					auto cb_next_block = read_integer(b & 31, m_stream);
					if (cb_next_block >= invalid_remaining)
						throw ill_formatted{};
					m_remaining_in_current_block = cb_next_block;
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
		{
			if (m_remaining_length == std::numeric_limits<uint64_t>::max())
				throw ill_formatted{};
		}
		array(Stream&& s)
			: m_stream(std::move(s))
			, m_remaining_length(std::numeric_limits<uint64_t>::max())
		{}
		using tag = tags::array;

		optional<document<stream::reader_ref_type_t<Stream>>> read()
		{
			if (m_remaining_length == 0)
				return nullopt;

			auto document = read_no_debug_check(stream::ref(m_stream));
			if (m_remaining_length != std::numeric_limits<uint64_t>::max())
				--m_remaining_length;
			if (!document)
			{
				if (m_remaining_length == std::numeric_limits<uint64_t>::max())
					m_remaining_length = 0;
				else
					throw ill_formatted{};
			}
			return document;
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
		map(Stream&& s, uint64_t remaining_length)
			: m_stream(std::move(s))
			, m_remaining_length(remaining_length)
		{
			if (m_remaining_length == std::numeric_limits<uint64_t>::max())
				throw ill_formatted{};
		}
		map(Stream&& s)
			: m_stream(std::move(s))
			, m_remaining_length(std::numeric_limits<uint64_t>::max())
		{}
		using tag = tags::map;

		optional<document<stream::reader_ref_type_t<Stream>>> read_key()
		{
			if (m_remaining_length == 0)
				return nullopt;

			auto document = read_no_debug_check(stream::ref(m_stream));
			if (!document)
			{
				if (m_remaining_length == std::numeric_limits<uint64_t>::max())
					m_remaining_length = 0;
				else
					throw ill_formatted{};
			}
			if (m_remaining_length != std::numeric_limits<uint64_t>::max())
				--m_remaining_length;
			return document;
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

	static_assert(sizeof(float) == sizeof(uint32_t), "Expect 32 bit floats");
	inline float to_float(uint32_t x) { return *reinterpret_cast<float*>(&x); }

	static_assert(sizeof(double) == sizeof(uint64_t), "Expect 64 bit doubles");
	inline double to_double(uint64_t x) { return *reinterpret_cast<double*>(&x); }

	// Read one integer from the stream. Additional is the low 5 bits of the first byte of the document
	// (ie the byte stripped of the major type)
	template <class Stream> uint64_t read_integer(uint8_t additional, Stream& s)
	{
		if (additional <= 23)
			return additional;
		else if (additional == 24)
			return read<uint8_t>(s);
		else if (additional == 25)
			return from_big_endian(read<uint16_t>(s));
		else if (additional == 26)
			return from_big_endian(read<uint32_t>(s));
		else if (additional == 27)
			return from_big_endian(read<uint64_t>(s));
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
		static optional<document<Stream>> fn_small_uint(Stream&& s, uint8_t first_byte) { return static_cast<uint64_t>(first_byte & 31); };
		static optional<document<Stream>> fn_uint_8(Stream&& s, uint8_t first_byte) { return static_cast<uint64_t>(stream::read<uint8_t>(s)); }
		static optional<document<Stream>> fn_uint_16(Stream&& s, uint8_t first_byte) { return static_cast<uint64_t>(from_big_endian(stream::read<uint16_t>(s))); }
		static optional<document<Stream>> fn_uint_32(Stream&& s, uint8_t first_byte) { return static_cast<uint64_t>(from_big_endian(stream::read<uint32_t>(s))); }
		static optional<document<Stream>> fn_uint_64(Stream&& s, uint8_t first_byte) { return from_big_endian(stream::read<uint64_t>(s)); }
		static optional<document<Stream>> fn_neg_int(Stream&& s, uint8_t first_byte)
		{
			auto x = read_integer(static_cast<uint8_t>(first_byte & 31), s);
			if (x > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
				throw ill_formatted{};

			return -1 - static_cast<int64_t>(x);
		}

		static optional<document<Stream>> fn_tag(Stream&& s, uint8_t first_byte)
		{
			// tags are skipped for now
			do
			{
				read_integer(static_cast<uint8_t>(first_byte & 31), s);
				first_byte = stream::read<uint8_t>(s);
			} while ((first_byte >> 5) == 6); // 6 is the major type for tags

			return read(std::forward<Stream>(s), first_byte);
		}
		static optional<document<Stream>> fn_false(Stream&&, uint8_t) { return false; }
		static optional<document<Stream>> fn_true(Stream&&, uint8_t) { return true; }
		static optional<document<Stream>> fn_null(Stream&&, uint8_t) { return nullptr; }
		static optional<document<Stream>> fn_undefined(Stream&&, uint8_t) { return tags::undefined{}; }
		static optional<document<Stream>> fn_end_of_structure(Stream&&, uint8_t) { return nullopt; }

		static optional<document<Stream>> fn_float_16(Stream&& s, uint8_t) { return read_half_point_float(s); }
		static optional<document<Stream>> fn_float_32(Stream&& s, uint8_t) { return double{ to_float(from_big_endian(stream::read<uint32_t>(s))) }; }
		static optional<document<Stream>> fn_float_64(Stream&& s, uint8_t) { return to_double(from_big_endian(stream::read<uint64_t>(s))); }
		static optional<document<Stream>> fn_ill_formatted(Stream&&, uint8_t) { throw ill_formatted{}; };

		static optional<document<Stream>> fn_small_binary(Stream&& s, uint8_t first_byte) { return byte_string<Stream>{ std::move(s), static_cast<uint8_t>(first_byte & 31) }; };
		static optional<document<Stream>> fn_8_binary(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_binary(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), from_big_endian(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_binary(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), from_big_endian(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_binary(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s), from_big_endian(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_null_terminated_binary(Stream&& s, uint8_t) { return byte_string<Stream>{ std::move(s) }; };
		
		static optional<document<Stream>> fn_small_text(Stream&& s, uint8_t first_byte) { return text_string<Stream>{ std::move(s), static_cast<uint8_t>(first_byte & 31) }; };
		static optional<document<Stream>> fn_8_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), from_big_endian(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), from_big_endian(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s), from_big_endian(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_null_terminated_text(Stream&& s, uint8_t) { return text_string<Stream>{ std::move(s) }; };

		static optional<document<Stream>> fn_small_array(Stream&& s, uint8_t first_byte) { return array<Stream>{ std::move(s), static_cast<uint8_t>(first_byte & 31) }; };
		static optional<document<Stream>> fn_8_array(Stream&& s, uint8_t) { return array<Stream>{ std::move(s), stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_array(Stream&& s, uint8_t) { return array<Stream>{ std::move(s), from_big_endian(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_array(Stream&& s, uint8_t) { return array<Stream>{ std::move(s), from_big_endian(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_array(Stream&& s, uint8_t) { return array<Stream>{ std::move(s), from_big_endian(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_null_terminated_array(Stream&& s, uint8_t) { return array<Stream>{ std::move(s) }; };

		static optional<document<Stream>> fn_small_map(Stream&& s, uint8_t first_byte) { return map<Stream>{ std::move(s), static_cast<uint8_t>(first_byte & 31) }; };
		static optional<document<Stream>> fn_8_map(Stream&& s, uint8_t) { return map<Stream>{ std::move(s), stream::read<uint8_t>(s) }; };
		static optional<document<Stream>> fn_16_map(Stream&& s, uint8_t) { return map<Stream>{ std::move(s), from_big_endian(stream::read<uint16_t>(s)) }; };
		static optional<document<Stream>> fn_32_map(Stream&& s, uint8_t) { return map<Stream>{ std::move(s), from_big_endian(stream::read<uint32_t>(s)) }; };
		static optional<document<Stream>> fn_64_map(Stream&& s, uint8_t) { return map<Stream>{ std::move(s), from_big_endian(stream::read<uint64_t>(s)) }; };
		static optional<document<Stream>> fn_null_terminated_map(Stream&& s, uint8_t) { return map<Stream>{ std::move(s) }; };

		static optional<document<Stream>> read(Stream&& s, uint8_t first_byte)
		{
			using fn = optional<document<Stream>> (*) (Stream&&, uint8_t);

			// This is a jump table for the CBOR parser
			// To help with branch prediction, it's good to not have too many different values in the table
			// However, it's also good to have simple functions for simple items.
			// This is why uint 0 and 1 are special cased, but not uint 17 say
			static fn functions[] = {
				// MAJOR TYPE 0
				fn_uint<0>   ,fn_uint<1>   ,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,
				fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,
				fn_small_uint,fn_small_uint,fn_small_uint,fn_small_uint,
				fn_uint_8,fn_uint_16,fn_uint_32,fn_uint_64,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_ill_formatted,

				// MAJOR TYPE 1
				fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,
				fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,
				fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,fn_neg_int,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,

				// MAJOR TYPE 2
				fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,
				fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,
				fn_small_binary,fn_small_binary,fn_small_binary,fn_small_binary,
				fn_8_binary,fn_16_binary,fn_32_binary,fn_64_binary,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_null_terminated_binary,

				// MAJOR TYPE 3
				fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,
				fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,fn_small_text,
				fn_small_text,fn_small_text,fn_small_text,fn_small_text,
				fn_8_text,fn_16_text,fn_32_text,fn_64_text,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_null_terminated_text,

				// MAJOR TYPE 4
				fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,
				fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,fn_small_array,
				fn_small_array,fn_small_array,fn_small_array,fn_small_array,
				fn_8_array,fn_16_array,fn_32_array,fn_64_array,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_null_terminated_array,

				// MAJOR TYPE 5
				fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,
				fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,fn_small_map,
				fn_small_map,fn_small_map,fn_small_map,fn_small_map,
				fn_8_map,fn_16_map,fn_32_map,fn_64_map,
				fn_ill_formatted,fn_ill_formatted,fn_ill_formatted,
				fn_null_terminated_map,

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
			return functions[first_byte](std::move(s), first_byte);
		}
	};
	template <class Stream> optional<document<std::decay_t<Stream>>> read_no_debug_check(Stream&& s)
	{
		static_assert(
			!std::is_trivially_move_constructible<std::decay_t<Stream>>::value ||
			std::is_trivially_move_constructible<document<std::decay_t<Stream>>>::value, "A cbor document on a trivially move constructible stream should be trivially move constructible");
		return read_helper<std::decay_t<Stream>>::read(std::forward<Stream>(s), read<uint8_t>(s));
	}

	template <class Stream, class error_handler> auto read(Stream&& s, error_handler e)
	{
		auto d = read_no_debug_check(std::forward<Stream>(s));
		if (!d)
			throw ill_formatted{};
		return debug_checks::add_read_checks(std::move(*d), e);
	}
	template <class Stream> auto read(Stream&& s) { return read(std::forward<Stream>(s), debug_checks::default_error_handler{}); }
}}