#pragma once

#include <exception>
#include "array_ref.h"
#include "debug_checks_writer.h"
#include <limits>
#include "numbers.h"
#include "sax_writer.h"
#include "stream.h"

namespace goldfish { namespace cbor
{
	namespace details
	{
		template <uint8_t major, class Stream> void write_integer(Stream& s, uint64_t x)
		{
			if (x <= 23)
			{
				stream::write(s, static_cast<uint8_t>((major << 5) | x));
			}
			else if (x <= std::numeric_limits<uint8_t>::max())
			{
				stream::write(s, static_cast<uint16_t>((major << 5) | 24 | (x << 8)));
			}
			else if (x <= std::numeric_limits<uint16_t>::max())
			{
				stream::write(s, static_cast<uint8_t>((major << 5) | 25));
				stream::write(s, byte_swap(static_cast<uint16_t>(x)));
			}
			else if (x <= std::numeric_limits<uint32_t>::max())
			{
				stream::write(s, static_cast<uint8_t>((major << 5) | 26));
				stream::write(s, byte_swap(static_cast<uint32_t>(x)));
			}
			else
			{
				stream::write(s, static_cast<uint8_t>((major << 5) | 27));
				stream::write(s, byte_swap(x));
			}
		}
	}

	template <class Stream> class stream_writer
	{
	public:
		stream_writer(Stream&& s)
			: m_stream(std::move(s))
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			m_stream.write_buffer(buffer);
		}
		void flush() { }
	private:
		Stream m_stream;
	};

	template <class Stream, uint8_t major> class indefinite_stream_writer
	{
	public:
		indefinite_stream_writer(Stream&& s)
			: m_stream(std::move(s))
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			details::write_integer<major>(m_stream, buffer.size());
			m_stream.write_buffer(buffer);
		}
		void flush()
		{
			stream::write(m_stream, static_cast<uint8_t>(0xFF));
		}
	private:
		Stream m_stream;
	};

	template <class Stream> class array_writer;
	template <class Stream> class indefinite_array_writer;
	template <class Stream> class map_writer;
	template <class Stream> class indefinite_map_writer;

	template <class Stream> class document_writer
	{
	public:
		document_writer(Stream&& s)
			: m_stream(std::move(s))
		{}
		void write(bool x)
		{
			if (x) stream::write(m_stream, static_cast<uint8_t>((7 << 5) | 21));
			else   stream::write(m_stream, static_cast<uint8_t>((7 << 5) | 20));
		}
		void write(nullptr_t)
		{
			stream::write(m_stream, static_cast<uint8_t>((7 << 5) | 22));
		}
		void write(double x)
		{
			stream::write(m_stream, static_cast<uint8_t>((7 << 5) | 27));
			auto i = *reinterpret_cast<uint64_t*>(&x);
			stream::write(m_stream, byte_swap(i));
		}
		void write(tags::undefined) 
		{
			stream::write(m_stream, static_cast<uint8_t>((7 << 5) | 23));
		}

		void write(uint64_t x) { details::write_integer<0>(m_stream, x); }
		void write(int64_t x) { details::write_integer<1>(m_stream, static_cast<uint64_t>(-1ll - x)); }
		stream_writer<Stream> write_binary(uint64_t cb)
		{
			details::write_integer<2>(m_stream, cb);
			return{ std::move(m_stream) };
		}
		stream_writer<Stream> write_text(uint64_t cb)
		{
			details::write_integer<3>(m_stream, cb);
			return{ std::move(m_stream) };
		}
		indefinite_stream_writer<Stream, 2> write_binary()
		{
			stream::write(m_stream, static_cast<uint8_t>((2 << 5) | 31));
			return{ std::move(m_stream) };
		}
		indefinite_stream_writer<Stream, 3> write_text()
		{
			stream::write(m_stream, static_cast<uint8_t>((3 << 5) | 31));
			return{ std::move(m_stream) };
		}

		array_writer<Stream> write_array(uint64_t size);
		indefinite_array_writer<Stream> write_array();

		map_writer<Stream> write_map(uint64_t size);
		indefinite_map_writer<Stream> write_map();

		template <class Document> std::enable_if_t<tags::has_tag<Document, tags::document>::value, void> write(Document& d)
		{
			copy_document(*this, d);
		}

	private:
		Stream m_stream;
	};
	template <class Stream> document_writer<std::decay_t<Stream>> write_no_debug_check(Stream&& s)
	{
		return{ std::forward<Stream>(s) };
	}
	template <class Stream, class error_handler> auto create_writer(Stream&& s, error_handler e)
	{
		return debug_checks::add_write_checks(write_no_debug_check(std::forward<Stream>(s)), e);
	}
	template <class Stream> auto create_writer(Stream&& s)
	{
		return create_writer(std::forward<Stream>(s), debug_checks::default_error_handler{});
	}

	template <class Stream> class array_writer
	{
	public:
		array_writer(Stream&& s)
			: m_stream(std::move(s))
		{}

		document_writer<stream::writer_ref_type_t<Stream>> append() { return{ stream::ref(m_stream) }; }
		void flush() {}
	private:
		Stream m_stream;
	};
	template <class Stream> class indefinite_array_writer
	{
	public:
		indefinite_array_writer(Stream& s)
			: m_stream(s)
		{}
		document_writer<stream::writer_ref_type_t<Stream>> append() { return{ stream::ref(m_stream) }; }
		void flush() { stream::write(m_stream, static_cast<uint8_t>(0xFF)); }
	private:
		Stream m_stream;
	};
	template <class Stream> array_writer<Stream> document_writer<Stream>::write_array(uint64_t size)
	{
		details::write_integer<4>(m_stream, size);
		return{ std::move(m_stream) };
	}
	template <class Stream> indefinite_array_writer<Stream> document_writer<Stream>::write_array()
	{
		stream::write(m_stream, static_cast<uint8_t>((4 << 5) | 31));
		return{ std::move(m_stream) };
	}

	template <class Stream> class map_writer
	{
	public:
		map_writer(Stream&& s)
			: m_stream(std::move(s))
		{}

		document_writer<stream::writer_ref_type_t<Stream>> append_key() { return{ stream::ref(m_stream) }; }
		document_writer<stream::writer_ref_type_t<Stream>> append_value() { return{ stream::ref(m_stream) }; }
		void flush() {}
	private:
		Stream m_stream;
	};
	template <class Stream> class indefinite_map_writer
	{
	public:
		indefinite_map_writer(Stream&& s)
			: m_stream(std::move(s))
		{}

		document_writer<stream::writer_ref_type_t<Stream>> append_key() { return{ stream::ref(m_stream) }; }
		document_writer<stream::writer_ref_type_t<Stream>> append_value() { return{ stream::ref(m_stream) }; }
		void flush() { stream::write(m_stream, static_cast<uint8_t>(0xFF)); }
	private:
		Stream m_stream;
	};
	template <class Stream> map_writer<Stream> document_writer<Stream>::write_map(uint64_t size)
	{
		details::write_integer<5>(m_stream, size);
		return{ std::move(m_stream) };
	}
	template <class Stream> indefinite_map_writer<Stream> document_writer<Stream>::write_map()
	{
		stream::write(m_stream, static_cast<uint8_t>((5 << 5) | 31));
		return{ std::move(m_stream) };
	}
}}