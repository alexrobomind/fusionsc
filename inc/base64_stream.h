#pragma once

#include "array_ref.h"
#include "stream.h"

namespace gold_fish { namespace stream
{ 
	// Reads binary data assuming inner reads base64
	template <class inner>
	class base64_reader
	{
	public:
		base64_reader(inner&& stream)
			: m_stream(std::move(stream))
		{}

		size_t read_buffer(buffer_ref data)
		{
			auto original_size = data.size();
			read_from_buffer(data);
			while (data.size() >= 3)
			{
				auto c_read = read_up_to_3(data.data());
				data.remove_front(c_read);
				if (c_read != 3)
					return original_size - data.size();
			}

			if (!data.empty())
			{
				m_cb_in_buffer = read_up_to_3(m_buffer.data());
				read_from_buffer(data);
			}
			return original_size - data.size();
		}
		auto& base() { return m_stream; }
	private:
		void read_from_buffer(buffer_ref& data)
		{
			auto cb_to_copy = static_cast<uint8_t>(std::min<size_t>(data.size(), m_cb_in_buffer));
			copy(const_buffer_ref{ m_buffer.data(), m_buffer.data() + cb_to_copy }, data.remove_front(cb_to_copy));
			m_cb_in_buffer -= cb_to_copy;
			std::copy(m_buffer.begin() + cb_to_copy, m_buffer.end(), m_buffer.begin());
		}
		uint8_t read_up_to_3(uint8_t* output)
		{
			// Read 4 characters, output 3
			uint8_t buffer[4] = { 0, 0, 0, 0 };
			auto c_read = m_stream.read_buffer(buffer);
			while (c_read && buffer[c_read - 1] == '=')
				--c_read;
			switch (c_read)
			{
			case 0: return 0;
			case 1: throw 0;
			case 2:
			{
				auto a = to_byte(buffer[0]);
				auto b = to_byte(buffer[1]);
				if (b & 0xF) throw 0;
				output[0] = ((a << 2) | (b >> 4));
				return 1;
			}
			case 3:
			{
				auto a = to_byte(buffer[0]);
				auto b = to_byte(buffer[1]);
				auto c = to_byte(buffer[2]);
				if (c & 0x3) throw 0;
				output[0] = ((a << 2) | (b >> 4));
				output[1] = (((b & 0xF) << 4) | (c >> 2));
				return 2;
			}
			case 4:
			{
				auto a = to_byte(buffer[0]);
				auto b = to_byte(buffer[1]);
				auto c = to_byte(buffer[2]);
				auto d = to_byte(buffer[3]);
				output[0] = ((a << 2) | (b >> 4));
				output[1] = (((b & 0xF) << 4) | (c >> 2));
				output[2] = (((c & 0x3) << 6) | d);
				return 3;
			}
			default: std::terminate();
			}
		}
		uint8_t to_byte(uint8_t c)
		{
			const uint8_t lookup_table[] = {
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
				throw 0;
			return result;
		}
		inner m_stream;
		std::array<uint8_t, 3> m_buffer;
		uint8_t m_cb_in_buffer = 0;
	};

	// Write base64 data to inner when binary data is provided
	template <class inner>
	class base64_writer : private assert_work_done
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
			if (m_cb_in_buffer == 1)
			{
				if (data.size() >= 2)
				{
					write_triplet(m_buffer[0], data[0], data[1]);
					data.remove_front(2);
					m_cb_in_buffer = 0;
				}
				else if (!data.empty())
				{
					m_buffer[1] = data.front();
					++m_cb_in_buffer;
					return;
				}
			}
			else if (m_cb_in_buffer == 2)
			{
				if (data.empty())
					return;

				write_triplet(m_buffer[0], m_buffer[1], data[0]);
				data.remove_front(1);
				m_cb_in_buffer = 0;
			}

			while (data.size() >= 3)
			{
				write_triplet(data[0], data[1], data[2]);
				data.remove_front(3);
			}

			std::copy(data.begin(), data.end(), m_buffer.begin());
			m_cb_in_buffer = static_cast<uint32_t>(data.size());
		}
		void flush()
		{
			if (m_cb_in_buffer == 1)
				write_triplet_flush(m_buffer[0]);
			else if (m_cb_in_buffer == 2)
				write_triplet_flush(m_buffer[0], m_buffer[1]);
			m_stream.flush();
			m_cb_in_buffer = 0;
			mark_work_done();
		}
		auto& base() { return m_stream; }
	private:
		uint8_t byte_for(uint8_t x)
		{
			static const char table[65] =
				"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
				"abcdefghijklmnopqrstuvwxyz"
				"0123456789+/";
			return static_cast<uint8_t>(table[x]);
		}
		void write_triplet(uint32_t a, uint32_t b, uint32_t c)
		{
			uint32_t x = (a << 16) | (b << 8) | c;
			m_stream.write(byte_for((x >> 18) & 63));
			m_stream.write(byte_for((x >> 12) & 63));
			m_stream.write(byte_for((x >> 6 ) & 63));
			m_stream.write(byte_for((x      ) & 63));
		}
		void write_triplet_flush(uint32_t a)
		{
			m_stream.write(byte_for((a >> 2) & 63));
			m_stream.write(byte_for((a & 3) << 4));
			m_stream.write('=');
			m_stream.write('=');
		}
		void write_triplet_flush(uint32_t a, uint32_t b)
		{
			uint32_t x = (a << 8) | b;
			m_stream.write(byte_for((x >> 10) & 63));
			m_stream.write(byte_for((x >> 4) & 63));
			m_stream.write(byte_for((x & 15) << 2));
			m_stream.write('=');
		}

		inner m_stream;
		std::array<uint8_t, 2> m_buffer;
		uint32_t m_cb_in_buffer = 0;
	};

	template <class inner> enable_if_reader_t<inner, base64_reader<std::decay_t<inner>>> base64(inner&& stream) { return{ std::forward<inner>(stream) }; }
	template <class inner> enable_if_writer_t<inner, base64_writer<std::decay_t<inner>>> base64(inner&& stream) { return{ std::forward<inner>(stream) }; }
}}