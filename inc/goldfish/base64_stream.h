#pragma once

#include "array_ref.h"
#include "stream.h"

namespace goldfish { namespace stream
{
	struct ill_formatted_base64_data : ill_formatted { using ill_formatted::ill_formatted; };

	// Reads binary data assuming inner reads base64
	template <class inner> class base64_reader
	{
	public:
		base64_reader(inner&& stream)
			: m_stream(std::move(stream))
		{}

		size_t read_partial_buffer(buffer_ref data)
		{
			auto original_size = data.size();
			read_from_already_parsed(data);
			while (data.size() >= 3)
			{
				auto c_read = deserialize_up_to_3_bytes(data);
				data.remove_front(c_read);
				if (c_read != 3)
					return original_size - data.size();
			}

			if (!data.empty())
			{
				assert(m_cb_already_parsed == 0); // because there is left over in data, read_from_already_parsed emptied m_already_parsed
				m_cb_already_parsed = deserialize_up_to_3_bytes(m_already_parsed);
				read_from_already_parsed(data);
			}
			return original_size - data.size();
		}
	private:
		void read_from_already_parsed(buffer_ref& data)
		{
			auto cb_to_copy = static_cast<uint8_t>(std::min<size_t>(data.size(), m_cb_already_parsed));
			copy(const_buffer_ref{ m_already_parsed.data(), m_already_parsed.data() + cb_to_copy }, data.remove_front(cb_to_copy));
			m_cb_already_parsed -= cb_to_copy;
			std::copy(m_already_parsed.begin() + cb_to_copy, m_already_parsed.end(), m_already_parsed.begin());
		}

		// Read up to 4 characters (or the end of stream), remove the potential padding (base64 can be padded with '=' characters at the end)
		// and generate up to 3 bytes of data
		uint8_t deserialize_up_to_3_bytes(buffer_ref output)
		{
			byte buffer[4];
			auto c_read = read_full_buffer(m_stream, buffer);
			if (c_read == 4 && buffer[3] == '=') // Presence of padding means the stream is made of blocks of 4 bytes
			{
				if (buffer[2] == '=')
					c_read = 2;
				else
					c_read = 3;

				if (stream::seek(m_stream, 1) != 0)
					throw ill_formatted_base64_data{ "'=' is only allowed at the end of a base64 stream" };
			}

			if (c_read == 0)
				return 0;
			if (c_read == 1)
				throw ill_formatted_base64_data{ "Unexpected number of characters in base64 stream" };

			auto a = character_to_6bits(buffer[0]);
			auto b = character_to_6bits(buffer[1]);
			output[0] = ((a << 2) | (b >> 4));
			if (c_read == 2)
			{
				if (b & 0xF)
					throw ill_formatted_base64_data{ "Invalid character at end of base64 stream" };
				return 1;
			}

			auto c = character_to_6bits(buffer[2]);
			output[1] = (((b & 0xF) << 4) | (c >> 2));
			if (c_read == 3)
			{
				if (c & 0x3)
					throw ill_formatted_base64_data{ "Invalid character at end of base64 stream" };
				return 2;
			}

			auto d = character_to_6bits(buffer[3]);
			output[2] = (((c & 0x3) << 6) | d);
			return 3;
		}
		byte character_to_6bits(byte c)
		{
			static const byte lookup_table[] = {
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,62,64,64,64,63,52,53,
				54,55,56,57,58,59,60,61,64,64,
				64,65,64,64,64, 0, 1, 2, 3, 4,
				 5, 6, 7, 8, 9,10,11,12,13,14,
				15,16,17,18,19,20,21,22,23,24,
				25,64,64,64,64,64,64,26,27,28,
				29,30,31,32,33,34,35,36,37,38,
				39,40,41,42,43,44,45,46,47,48,
				49,50,51,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64,64,64,64,64,
				64,64,64,64,64,64
			};
			static_assert(sizeof(lookup_table) == 256, "");
			auto result = lookup_table[c];
			if (result >= 64)
				throw ill_formatted_base64_data{ "Invalid character in base64 stream" };
			return result;
		}
		inner m_stream;
		std::array<byte, 3> m_already_parsed;
		uint8_t m_cb_already_parsed = 0;
	};

	// Write base64 data to inner when binary data is provided
	template <class inner> class base64_writer
	{
	public:
		base64_writer(inner&& stream)
			: m_stream(std::move(stream))
		{}
		base64_writer(base64_writer&&) = default;
		base64_writer(const base64_writer&) = delete;
		base64_writer& operator = (const base64_writer&) = delete;

		void write_buffer(const_buffer_ref data)
		{
			if (data.empty())
				return;

			if (m_cb_pending_encoding == 1)
			{
				if (data.size() >= 2)
				{
					write_triplet(m_pending_encoding[0], data[0], data[1]);
					data.remove_front(2);
					m_cb_pending_encoding = 0;
				}
				else
				{
					m_pending_encoding[1] = data.front();
					++m_cb_pending_encoding;
					return;
				}
			}
			else if (m_cb_pending_encoding == 2)
			{
				write_triplet(m_pending_encoding[0], m_pending_encoding[1], data[0]);
				data.remove_front(1);
				m_cb_pending_encoding = 0;
			}

			while (data.size() >= 3)
			{
				write_triplet(data[0], data[1], data[2]);
				data.remove_front(3);
			}

			std::copy(data.begin(), data.end(), m_pending_encoding.begin());
			m_cb_pending_encoding = static_cast<uint8_t>(data.size());
		}
		auto flush()
		{
			flush_no_inner_stream_flush();
			return m_stream.flush();
		}
		void flush_no_inner_stream_flush()
		{
			if (m_cb_pending_encoding == 1)
				write_triplet_flush(m_pending_encoding[0]);
			else if (m_cb_pending_encoding == 2)
				write_triplet_flush(m_pending_encoding[0], m_pending_encoding[1]);
			m_cb_pending_encoding = 0;
		}
		auto& inner_stream() { return m_stream; }
	private:
		byte character_from_6bits(byte x)
		{
			static const char table[65] =
				"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
				"abcdefghijklmnopqrstuvwxyz"
				"0123456789+/";
			return static_cast<byte>(table[x]);
		}
		void write_triplet(uint32_t a, uint32_t b, uint32_t c)
		{
			uint32_t x = (a << 16) | (b << 8) | c;
			stream::write(m_stream, character_from_6bits((x >> 18) & 63));
			stream::write(m_stream, character_from_6bits((x >> 12) & 63));
			stream::write(m_stream, character_from_6bits((x >> 6 ) & 63));
			stream::write(m_stream, character_from_6bits((x      ) & 63));
		}
		void write_triplet_flush(uint32_t a)
		{
			stream::write(m_stream, character_from_6bits((a >> 2) & 63));
			stream::write(m_stream, character_from_6bits((a & 3) << 4));
			stream::write(m_stream, '=');
			stream::write(m_stream, '=');
		}
		void write_triplet_flush(uint32_t a, uint32_t b)
		{
			uint32_t x = (a << 8) | b;
			stream::write(m_stream, character_from_6bits((x >> 10) & 63));
			stream::write(m_stream, character_from_6bits((x >> 4) & 63));
			stream::write(m_stream, character_from_6bits((x & 15) << 2));
			stream::write(m_stream, '=');
		}

		inner m_stream;
		std::array<byte, 2> m_pending_encoding;
		uint8_t m_cb_pending_encoding = 0;
	};

	template <class inner> enable_if_reader_t<inner, base64_reader<std::decay_t<inner>>> decode_base64(inner&& stream) { return{ std::forward<inner>(stream) }; }
	template <class inner> enable_if_writer_t<inner, base64_writer<std::decay_t<inner>>> encode_base64_to(inner&& stream) { return{ std::forward<inner>(stream) }; }
}}