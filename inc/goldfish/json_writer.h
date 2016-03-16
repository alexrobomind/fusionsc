#pragma once

#include <string>
#include "array_ref.h"
#include "base64_stream.h"
#include "debug_checks_writer.h"
#include "dom.h"
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
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			for (auto&& c : buffer)
			{
				switch (c)
				{
				case '\b': stream::write(m_stream, '\\'); stream::write(m_stream, 'b'); break;
				case '\n': stream::write(m_stream, '\\'); stream::write(m_stream, 'n'); break;
				case '\r': stream::write(m_stream, '\\'); stream::write(m_stream, 'r'); break;
				case '\t': stream::write(m_stream, '\\'); stream::write(m_stream, 't'); break;

				case '"':
				case '\\':
					stream::write(m_stream, '\\');
					stream::write(m_stream, static_cast<char>(c));
					break;

				default:
					if (c < 0x20)
					{
						char data[6] = { '\\', 'u', '0', '0', '0', '0' };
						data[4] = "0123456789ABCDEF"[c >> 4];
						data[5] = "0123456789ABCDEF"[c & 15];
						m_stream.write_buffer({ reinterpret_cast<const uint8_t*>(data), 6 });
					}
					else
					{
						stream::write(m_stream, c);
					}
					break;
				}
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
		{}
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
		{}

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
		{}

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
			if (x) m_stream.write_buffer({ reinterpret_cast<const uint8_t*>("true"), 4 });
			else   m_stream.write_buffer({ reinterpret_cast<const uint8_t*>("false"), 5 });
			return m_stream.flush();
		}
		auto write(nullptr_t)
		{
			m_stream.write_buffer({ reinterpret_cast<const uint8_t*>("null"), 4 });
			return m_stream.flush();
		}
		auto write(tags::undefined)
		{
			return write(nullptr);
		}
		auto write(uint64_t x)
		{
			auto string = std::to_string(x);
			m_stream.write_buffer({ reinterpret_cast<const uint8_t*>(string.data()), string.size() });
			return m_stream.flush();
		}
		auto write(int64_t x)
		{
			auto string = std::to_string(x);
			m_stream.write_buffer({ reinterpret_cast<const uint8_t*>(string.data()), string.size() });
			return m_stream.flush();
		}
		auto write(double x)
		{
			char buffer[1024];
			auto cb = sprintf_s(buffer, "%g", x);
			if (cb <= 0)
				std::terminate();
			m_stream.write_buffer({ reinterpret_cast<const uint8_t*>(buffer), static_cast<size_t>(cb) });
			return m_stream.flush();
		}

		auto start_binary(uint64_t cb) { return start_binary(); }
		auto start_string(uint64_t cb) { return start_string(); }
		binary_writer<Stream> start_binary()
		{
			stream::write(m_stream, '"');
			return{ std::move(m_stream) };
		}
		text_writer<Stream> start_string()
		{
			stream::write(m_stream, '"');
			return{ std::move(m_stream) };
		}

		auto start_array(uint64_t size) { return start_array(); }
		array_writer<Stream> start_array()
		{
			stream::write(m_stream, '[');
			return{ std::move(m_stream) };
		}
		
		auto start_map(uint64_t size) { return start_map(); }
		map_writer<Stream> start_map()
		{
			stream::write(m_stream, '{');
			return{ std::move(m_stream) };
		}

	private:
		Stream m_stream;
	};
	template <class Stream> document_writer<std::decay_t<Stream>> write_no_debug_check(Stream&& s) { return{ std::forward<Stream>(s) }; }
	template <class Stream, class error_handler> auto create_writer(Stream&& s, error_handler e)
	{
		return sax::make_writer(debug_checks::add_write_checks(write_no_debug_check(std::forward<Stream>(s)), e));
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