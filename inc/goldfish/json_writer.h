#pragma once

#include <string>
#include "array_ref.h"
#include "base64_stream.h"
#include "debug_checks_writer.h"
#include "sax_writer.h"
#include "stream.h"

namespace goldfish { namespace json
{
	struct ill_formatted_utf8 {};
	template <class Stream> class document_writer;

	template <class Stream> class text_writer
	{
	public:
		text_writer(Stream&& s)
			: m_stream(std::move(s))
		{
			stream::write(m_stream, '"');
		}
		void write_buffer(const_buffer_ref buffer)
		{
			enum category : uint8_t
			{
				F, // forward to inner stream
				B, // \b
				N, // \n
				R, // \r
				T, // \t
				Q, // \"
				S, // \\ 
				U, // \u00??
			};
			static const category lookup[] = {
				/*       0 1 2 3 4 5 6 7 8 9 A B C D E F */
				/*0x00*/ U,U,U,U,U,U,U,U,B,T,N,U,U,R,U,U,
				/*0x10*/ U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,
				/*0x20*/ F,F,Q,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0x30*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0x40*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0x50*/ F,F,F,F,F,F,F,F,F,F,F,F,S,F,F,F,
				/*0x60*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0x70*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0x80*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0x90*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0xA0*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0xB0*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0xC0*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0xD0*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0xE0*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
				/*0xF0*/ F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
			};
			static_assert(sizeof(lookup) / sizeof(lookup[0]) == 256, "");

			if (buffer.empty())
				return;

			auto it = buffer.begin();
			for (;;)
			{
				auto prev = it;
				while (it != buffer.end() && lookup[*it] == F)
					++it;
				m_stream.write_buffer({ prev, it });
				if (it == buffer.end())
					break;

				switch (lookup[*it])
				{
					case B: stream::write(m_stream, '\\'); stream::write(m_stream, 'b'); break;
					case N: stream::write(m_stream, '\\'); stream::write(m_stream, 'n'); break;
					case R: stream::write(m_stream, '\\'); stream::write(m_stream, 'r'); break;
					case T: stream::write(m_stream, '\\'); stream::write(m_stream, 't'); break;
					case Q: stream::write(m_stream, '\\'); stream::write(m_stream, '"'); break;
					case S: stream::write(m_stream, '\\'); stream::write(m_stream, '\\'); break;
					case U: 
					{
						char data[6] = { '\\', 'u', '0', '0', '0', '0' };
						data[4] = "0123456789ABCDEF"[*it >> 4];
						data[5] = "0123456789ABCDEF"[*it & 15];
						m_stream.write_buffer({ reinterpret_cast<const byte*>(data), 6 });
					}
					break;
				}
				++it;
				if (it == buffer.end())
					break;
			}
		}
		auto flush()
		{
			stream::write(m_stream, '"');
			return m_stream.flush();
		}
	private:
		Stream m_stream;
	};

	template <class Stream> class binary_writer
	{
	public:
		binary_writer(Stream&& s)
			: m_stream(std::move(s))
		{
			stream::write(m_stream.inner_stream(), '"');
		}
		void write_buffer(const_buffer_ref buffer)
		{
			m_stream.write_buffer(buffer);
		}
		auto flush()
		{
			m_stream.flush_no_inner_stream_flush();
			stream::write(m_stream.inner_stream(), '"');
			return m_stream.inner_stream().flush();
		}
	private:
		stream::base64_writer<Stream> m_stream;
	};

	template <class Stream> class array_writer
	{
	public:
		array_writer(Stream&& s)
			: m_stream(std::move(s))
		{
			stream::write(m_stream, '[');
		}

		document_writer<stream::writer_ref_type_t<Stream>> append();
		auto flush()
		{
			stream::write(m_stream, ']');
			return m_stream.flush();
		}
	private:
		Stream m_stream;
		bool m_first = true;
	};

	template <class Stream> class map_writer
	{
	public:
		map_writer(Stream&& s)
			: m_stream(std::move(s))
		{
			stream::write(m_stream, '{');
		}

		document_writer<stream::writer_ref_type_t<Stream>> append_key();
		document_writer<stream::writer_ref_type_t<Stream>> append_value();
		auto flush()
		{
			stream::write(m_stream, '}');
			return m_stream.flush();
		}
	private:
		Stream m_stream;
		bool m_first = true;
	};

	template <class Stream> class document_writer
	{
	public:
		document_writer(Stream&& s)
			: m_stream(std::move(s))
		{}
		auto write(bool x)
		{
			if (x) m_stream.write_buffer({ reinterpret_cast<const byte*>("true"), 4 });
			else   m_stream.write_buffer({ reinterpret_cast<const byte*>("false"), 5 });
			return m_stream.flush();
		}
		auto write(nullptr_t)
		{
			m_stream.write_buffer({ reinterpret_cast<const byte*>("null"), 4 });
			return m_stream.flush();
		}
		auto write(tags::undefined)
		{
			return write(nullptr);
		}
		auto write(uint64_t x)
		{
			if (x < 10)
			{
				stream::write(m_stream, static_cast<char>('0' + x));
				return m_stream.flush();
			}

			uint8_t buffer[20];
			uint8_t* it = buffer;
			while (x != 0)
			{
				*it++ = '0' + (x % 10);
				x /= 10;
			}
			std::reverse(buffer, it);
			m_stream.write_buffer({ buffer, it });
			return m_stream.flush();
		}
		auto write(int64_t x)
		{
			if (x < 0)
			{
				stream::write(m_stream, '-');
				return write(static_cast<uint64_t>(-x));
			}
			else
			{
				return write(static_cast<uint64_t>(x));
			}
		}
		auto write(double x)
		{
			auto string = std::to_string(x);
			m_stream.write_buffer({ reinterpret_cast<const byte*>(string.data()), string.size() });
			return m_stream.flush();
		}

		auto start_binary(uint64_t cb) { return start_binary(); }
		auto start_string(uint64_t cb) { return start_string(); }
		binary_writer<Stream> start_binary() { return{ std::move(m_stream) }; }
		text_writer<Stream> start_string() { return{ std::move(m_stream) }; }

		auto start_array(uint64_t size) { return start_array(); }
		array_writer<Stream> start_array() { return{ std::move(m_stream) }; }
		
		auto start_map(uint64_t size) { return start_map(); }
		map_writer<Stream> start_map() { return{ std::move(m_stream) }; }

	private:
		Stream m_stream;
	};
	template <class Stream> document_writer<std::decay_t<Stream>> create_writer_no_debug_check(Stream&& s) { return{ std::forward<Stream>(s) }; }
	template <class Stream, class error_handler> auto create_writer(Stream&& s, error_handler e)
	{
		return sax::make_writer(debug_checks::add_write_checks(create_writer_no_debug_check(std::forward<Stream>(s)), e));
	}
	template <class Stream> auto create_writer(Stream&& s)
	{
		return create_writer(std::forward<Stream>(s), debug_checks::default_error_handler{});
	}

	template <class Stream> document_writer<stream::writer_ref_type_t<Stream>> array_writer<Stream>::append()
	{
		if (m_first)
			m_first = false;
		else
			stream::write(m_stream, ',');

		return{ stream::ref(m_stream) };
	}

	template <class Stream> document_writer<stream::writer_ref_type_t<Stream>> map_writer<Stream>::append_key()
	{
		if (m_first)
			m_first = false;
		else
			stream::write(m_stream, ',');
		return{ stream::ref(m_stream) };
	}
	template <class Stream> document_writer<stream::writer_ref_type_t<Stream>> map_writer<Stream>::append_value()
	{
		stream::write(m_stream, ':');
		return{ stream::ref(m_stream) };
	}
}}