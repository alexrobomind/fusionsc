#pragma once

#include "base64_stream.h"
#include "debug_checks_reader.h"
#include "tags.h"
#include "variant.h"
#include "stream.h"
#include "optional.h"
#include "sax_reader.h"

namespace goldfish { namespace json
{
	struct ill_formatted_json_data : ill_formatted { using ill_formatted::ill_formatted; };
	struct integer_overflow_in_json : ill_formatted_json_data { using ill_formatted_json_data::ill_formatted_json_data; };

	class byte_string;
	template <class Stream> class text_string;
	template <class Stream> class array;
	template <class Stream> class map;
	template <class Stream> struct document : document_impl<
		true /*does_json_conversions*/,
		bool,
		nullptr_t,
		uint64_t,
		int64_t,
		double,
		undefined,
		text_string<Stream>,
		byte_string,
		array<Stream>,
		map<Stream>>
	{
		using document_impl::document_impl;
	};
	template <class Stream> document<std::decay_t<Stream>> read_no_debug_check(Stream&& s);

	namespace details
	{
		template <class Stream> optional<char> peek_non_space(Stream& s)
		{
			for (;;)
			{
				auto c = s.peek<char>();
				if (c != ' ' && c != '\t' && c != '\r' && c != '\n')
					return c;
				else
					stream::read<char>(s);
			}
		}
		template <class Stream> char read_non_space(Stream& s)
		{
			for (;;)
			{
				auto c = stream::read<char>(s);
				if (c != ' ' && c != '\t' && c != '\r' && c != '\n')
					return c;
			}
		}
		template <class Stream> void throw_if_stream_isnt(Stream& s, std::initializer_list<char> string)
		{
			for (auto c : string)
			{
				if (stream::read<char>(s) != c)
					throw ill_formatted_json_data{ "Unexpected JSON document value" };
			}
		}
	}

	class byte_string
	{
	public:
		using tag = tags::binary;
		size_t read_partial_buffer(buffer_ref) { std::terminate(); }
	};

	/*
	Represents a stream that ends when the end quote is found in the JSON document
	The stream provided is expected to be right after the opening quote (the quote should have
	already been read).
	*/
	template <class Stream> class text_string
	{
	public:
		text_string(Stream&& s)
			: m_stream(std::move(s))
		{}
		using tag = tags::string;

		size_t read_partial_buffer(buffer_ref buffer)
		{
			if (m_converted.front() == end_of_stream)
				return 0;

			auto original = buffer.size();

			copy_from_converted(buffer);

			enum category : uint8_t
			{
				S, // simple (just needs to be forwarded to the inner stream)
				E, // escape: \ character
				Q, // quote: " character
				I, // character should have been escaped or is not a valid UTF8 character
			};
			static const category lookup[] = {
				/*       0 1 2 3 4 5 6 7 8 9 A B C D E F */
				/*0x00*/ I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,
				/*0x10*/ I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,
				/*0x20*/ S,S,Q,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0x30*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0x40*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0x50*/ S,S,S,S,S,S,S,S,S,S,S,S,E,S,S,S,
				/*0x60*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0x70*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0x80*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0x90*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0xA0*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0xB0*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0xC0*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0xD0*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0xE0*/ S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,
				/*0xF0*/ S,S,S,S,S,S,S,S,I,I,I,I,I,I,I,I,
			};
			static_assert(sizeof(lookup) / sizeof(lookup[0]) == 256, "The lookup table should have 256 entries");

			while (!buffer.empty())
			{
				byte c;
				auto it = buffer.begin();
				while (lookup[c = stream::read<byte>(m_stream)] == S)
				{
					*(it++) = c;
					if (it == buffer.end())
						return original;
				}
				buffer.remove_front(it - buffer.begin());

				switch (lookup[c])
				{
				case E:
					switch (stream::read<byte>(m_stream))
					{
					case '"':  buffer.pop_front() = '"';  break;
					case '\\': buffer.pop_front() = '\\'; break;
					case '/':  buffer.pop_front() = '/';  break;
					case 'b':  buffer.pop_front() = '\b'; break;
					case 'f':  buffer.pop_front() = '\f'; break;
					case 'n':  buffer.pop_front() = '\n'; break;
					case 'r':  buffer.pop_front() = '\r'; break;
					case 't':  buffer.pop_front() = '\t'; break;
					case 'u':
					{
						auto converted = compute_converted(read_utf32_character());
						std::copy(converted.begin() + 1, converted.end(), m_converted.begin());
						buffer.pop_front() = converted.front();
						copy_from_converted(buffer);
						break;
					}
					default: throw ill_formatted_json_data("Invalid escape sequence in JSON string");
					}
					break;

				case Q:
					m_converted.front() = end_of_stream; // Indicate we reached the end
					return original - buffer.size();

				default:
					throw ill_formatted_json_data{ "Invalid character in JSON string" };
				}
			}
			return original - buffer.size();
		}

	private:
		static const byte invalid_char = 0xFF;
		static const byte end_of_stream = 0xFE;
		void copy_from_converted(buffer_ref& buffer)
		{
			while (m_converted.front() != invalid_char && !buffer.empty())
			{
				assert(m_converted.front() != end_of_stream);

				buffer.pop_front() = m_converted.front();
				std::copy(m_converted.begin() + 1, m_converted.end(), m_converted.begin());
				m_converted.back() = invalid_char;
			}
		}
		static uint8_t parse_hex(char c)
		{
			if ('0' <= c && c <= '9') return c - '0';
			else if ('a' <= c && c <= 'f') return c - 'a' + 10;
			else if ('A' <= c && c <= 'F') return c - 'A' + 10;
			else throw ill_formatted_json_data{ "Invalid hexadecimal digit" };
		}
		uint16_t read_utf16_character()
		{
			uint16_t value = 0;
			value = (value << 4) | parse_hex(stream::read<char>(m_stream));
			value = (value << 4) | parse_hex(stream::read<char>(m_stream));
			value = (value << 4) | parse_hex(stream::read<char>(m_stream));
			value = (value << 4) | parse_hex(stream::read<char>(m_stream));
			return value;
		}
		uint32_t read_utf32_character()
		{
			uint32_t a = read_utf16_character();
			if (0xD800 <= a && a <= 0xDFFF)
			{
				if (a > 0xDBFF)
					throw ill_formatted_json_data{ "Invalid UTF32 encoding" };

				// We need a second character
				if (stream::read<char>(m_stream) != '\\') throw ill_formatted_json_data{ "Invalid serialization of surrogate pair" };
				if (stream::read<char>(m_stream) != 'u') throw ill_formatted_json_data{ "Invalid serialization of surrogate pair" };
				uint32_t b = read_utf16_character();
				if (b < 0xDC00 || b > 0xDFFF)
					throw ill_formatted_json_data{ "Invalid serialization of surrogate pair" };

				a -= 0xD800;
				b -= 0xDC00;
				return 0x10000 | (a << 10) | b;
			}
			else
			{
				return a;
			}
		}
		std::array<byte, 4> compute_converted(uint32_t codepoint)
		{
			auto get_6_bits = [&](uint32_t codepoint, int offset)
			{
				return static_cast<byte>(0b10000000 | ((codepoint >> offset) & 0b111111));
			};

			if (codepoint <= 0x7F)
				return { static_cast<byte>(codepoint), invalid_char, invalid_char, invalid_char };
			else if (codepoint <= 0x7FF)
				return{ static_cast<byte>(0b11000000 | (codepoint >> 6)), get_6_bits(codepoint, 0), invalid_char, invalid_char };
			else if (codepoint <= 0xFFFF)
				return{ static_cast<byte>(0b11100000 | (codepoint >> 12)), get_6_bits(codepoint, 6), get_6_bits(codepoint, 0), invalid_char };
			else if (codepoint <= 0x10FFFF)
				return{ static_cast<byte>(0b11110000 | (codepoint >> 18)), get_6_bits(codepoint, 12), get_6_bits(codepoint, 6), get_6_bits(codepoint, 0) };
			else
				throw ill_formatted_json_data{ "Invalid UTF32 encoding" };
		}

		Stream m_stream;

		// When parsing a unicode character encoded as \u????, we might generate up to 4 UTF-8 characters, but we won't necessarily output them all to the caller
		// This buffer keeps characters that have already been parsed but not output yet
		// The characters are "popped" from the front of the array, if they are not invalid
		// Once the end of the text is found (the unescaped " character is read), we write end_of_stream (0xFE) in this array
		std::array<byte, 3> m_converted{ invalid_char, invalid_char, invalid_char };
	public:
		// This member is used by variant to store the type info
		// This helps lower the size of a variant that contains a text_string by allowing variant to store the type in the padding rather than appending a new field
		uint8_t padding_for_variant;
	};

	template <class Stream, char end_character> class comma_separated_reader
	{
	public:
		comma_separated_reader(Stream&& s)
			: m_stream(std::move(s))
		{}
		optional<document<stream::reader_ref_type_t<Stream>>> read_comma_separated()
		{
			switch (m_state)
			{
				case state::first:
				{
					auto c = details::peek_non_space(m_stream);
					if (c == nullopt)
						throw stream::unexpected_end_of_stream{};

					if (c == end_character)
					{
						stream::read<char>(m_stream);
						m_state = state::ended;
						return nullopt;
					}
					else
					{
						m_state = state::middle;
						return read_no_debug_check(stream::ref(m_stream));
					}
				}

				case state::middle:
				{
					switch (details::read_non_space(m_stream))
					{
					case ',': return read_no_debug_check(stream::ref(m_stream));
					case end_character: m_state = state::ended; return nullopt;
					default: throw ill_formatted_json_data{ "Invalid delimiter in JSON array or map" };
					}
				}

				case state::ended:
					return nullopt;

				default: std::terminate();
			}
		}

		Stream m_stream;
		enum class state : uint8_t
		{
			first,
			middle,
			ended,
		} m_state = state::first;
	public:
		// This member is used by variant to store the type info
		// This helps lower the size of a variant that contains an array or a map by allowing variant to store the type in the padding rather than appending a new field
		uint8_t padding_for_variant;
	};
	template <class Stream> class array : public comma_separated_reader<Stream, ']'>
	{
	public:
		using tag = tags::array;
		using comma_separated_reader<Stream, ']'>::comma_separated_reader;
		auto read() { return read_comma_separated(); }
	};
	template <class Stream> class map : public comma_separated_reader<Stream, '}'>
	{
	public:
		using tag = tags::map;
		using comma_separated_reader<Stream, '}'>::comma_separated_reader;

		auto read_key()
		{
			auto key = read_comma_separated();
			if (key && !key->is_exactly<tags::string>())
				throw ill_formatted_json_data{ "Only strings are supported for JSON keys" };
			return key;
		}
		document<stream::reader_ref_type_t<Stream>> read_value()
		{
			if (details::read_non_space(m_stream) != ':')
				throw ill_formatted_json_data{ "':' expected between JSON key and value" };
			return read_no_debug_check(stream::ref(m_stream));
		}
	};

	template <class Stream> uint64_t read_unsigned_integer(Stream& s, char first, bool allow_leading_zeroes)
	{
		if (allow_leading_zeroes)
		{
			if (first < '0' || first > '9')
				throw ill_formatted_json_data{ "Invalid digit in JSON integer" };
		}
		else
		{
			if (first == '0')
				return 0u;

			if (first < '1' || first > '9')
				throw ill_formatted_json_data{ "Invalid digit in JSON integer" };
		}

		uint64_t result = (first - '0');
		for (;;)
		{
			auto c = s.peek<char>();
			if (c == nullopt || *c < '0' || *c > '9')
				return result;

			if (result > (std::numeric_limits<uint64_t>::max() - (*c - '0')) / 10)
				throw integer_overflow_in_json{ "JSON integer too large" };
			result = (result * 10) + *c - '0';
			stream::read<char>(s);
		}
	}
	template <class Stream> double read_decimals(Stream& s)
	{
		auto first = s.peek<char>();
		if (first == nullopt || *first < '0' || *first > '9')
			throw ill_formatted_json_data{ "Invalid digit in JSON integer" };

		double result = 0;
		double divider = 1;
		for (;;)
		{
			auto c = s.peek<char>();
			if (c == nullopt || *c < '0' || *c > '9')
				return result;

			divider *= 10;
			result += (*c - '0') / divider;
			stream::read<char>(s);
		}
	}
	template <class Stream> variant<uint64_t, int64_t, double> read_number(Stream& s, char first)
	{
		bool negative = false;
		if (first == '-')
		{
			negative = true;
			first = stream::read<char>(s);
		}

		auto integer = read_unsigned_integer(s, first, false /*allow_leading_zeroes*/);

		auto floating_point_marker = s.peek<char>();
		if (floating_point_marker != '.' && floating_point_marker != 'e' && floating_point_marker != 'E')
		{
			if (negative)
			{
				static_assert(std::numeric_limits<int64_t>::min() + 1 == -std::numeric_limits<int64_t>::max(),
					"our overflow check relies on int64_t range to be [-2^63 .. 2^63-1]");
				if (integer > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1)
					throw integer_overflow_in_json{ "JSON integer too large" };
				return -static_cast<int64_t>(integer);
			}
			else
			{
				return integer;
			}
		}

		double decimals = 0;
		if (floating_point_marker == '.')
		{
			stream::read<char>(s);
			decimals = read_decimals(s);
			floating_point_marker = s.peek<char>();
		}

		double multiplier = 1.;
		if (floating_point_marker == 'e' || floating_point_marker == 'E')
		{
			stream::read<char>(s);
			first = stream::read<char>(s);
			bool negative_exponent = false;
			if (first == '+' || first == '-')
			{
				negative_exponent = (first == '-');
				first = stream::read<char>(s);
			}
			auto exponent_value = read_unsigned_integer(s, first, true /*allow_leading_zeroes*/);
			multiplier = pow(10., exponent_value);
			if (negative_exponent)
				multiplier = 1 / multiplier;
		}

		return (negative ? -1 : 1) * multiplier * ((double)integer + decimals);
	}

	template <class Stream> document<std::decay_t<Stream>> read_no_debug_check(Stream&& s)
	{
		auto c = details::read_non_space(s);

		switch (c)
		{
			case '[': return array<std::decay_t<Stream>>{ std::forward<Stream>(s) };
			case '{': return map<std::decay_t<Stream>>{ std::forward<Stream>(s) };
			case 't': details::throw_if_stream_isnt(s, { 'r', 'u', 'e' }); return true;
			case 'f': details::throw_if_stream_isnt(s, { 'a', 'l', 's', 'e' }); return false;
			case 'n': details::throw_if_stream_isnt(s, { 'u', 'l', 'l' }); return nullptr;
			case '"': return text_string<std::decay_t<Stream>>{ std::forward<Stream>(s) };
			case '-':
			case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
				return read_number(s, c).visit([](auto&& x) -> document<Stream> { return x; });

			default: throw ill_formatted_json_data{ "Invalid first character for JSON document" };
		}
	}

	template <class Stream, class error_handler> auto read(Stream&& s, error_handler e)
	{
		return debug_checks::add_read_checks(read_no_debug_check(std::forward<Stream>(s)), e);
	}
	template <class Stream> auto read(Stream&& s) { return read(std::forward<Stream>(s), debug_checks::default_error_handler{}); }
}}