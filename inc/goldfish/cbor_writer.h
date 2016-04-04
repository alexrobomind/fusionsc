#pragma once

#include <exception>
#include "array_ref.h"
#include "common.h"
#include "debug_checks_writer.h"
#include <limits>
#include "sax_writer.h"
#include "stream.h"

namespace goldfish { namespace cbor
{
	namespace details
	{
		template <byte major, class Stream> void write_integer(Stream& s, uint64_t x)
		{
			if (x <= 23)
			{
				stream::write(s, static_cast<byte>((major << 5) | x));
			}
			else if (x <= std::numeric_limits<uint8_t>::max())
			{
				stream::write(s, static_cast<byte>((major << 5) | 24));
				stream::write(s, static_cast<uint8_t>(x));
			}
			else if (x <= std::numeric_limits<uint16_t>::max())
			{
				stream::write(s, static_cast<byte>((major << 5) | 25));
				stream::write(s, to_big_endian(static_cast<uint16_t>(x)));
			}
			else if (x <= std::numeric_limits<uint32_t>::max())
			{
				stream::write(s, static_cast<byte>((major << 5) | 26));
				stream::write(s, to_big_endian(static_cast<uint32_t>(x)));
			}
			else
			{
				stream::write(s, static_cast<byte>((major << 5) | 27));
				stream::write(s, to_big_endian(x));
			}
		}
	}
	
	template <class Stream, byte major> class indefinite_stream_writer
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
		auto flush()
		{
			stream::write(m_stream, static_cast<byte>(0xFF));
			return m_stream.flush();
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
		auto write(bool x)
		{
			if (x) stream::write(m_stream, static_cast<byte>((7 << 5) | 21));
			else   stream::write(m_stream, static_cast<byte>((7 << 5) | 20));
			return m_stream.flush();
		}
		auto write(nullptr_t)
		{
			stream::write(m_stream, static_cast<byte>((7 << 5) | 22));
			return m_stream.flush();
		}
		auto write(double x)
		{
			if (static_cast<float>(x) == x)
				return write(static_cast<float>(x));

			static_assert(sizeof(double) == sizeof(uint64_t), "Expect 64 bit doubles");
			stream::write(m_stream, static_cast<byte>((7 << 5) | 27));
			auto i = *reinterpret_cast<uint64_t*>(&x);
			stream::write(m_stream, to_big_endian(i));
			return m_stream.flush();
		}
		auto write(float x)
		{
			static_assert(sizeof(float) == sizeof(uint32_t), "Expect 32 bit floats");
			stream::write(m_stream, static_cast<byte>((7 << 5) | 26));
			auto i = *reinterpret_cast<uint32_t*>(&x);
			stream::write(m_stream, to_big_endian(i));
			return m_stream.flush();
		}
		auto write(undefined) 
		{
			stream::write(m_stream, static_cast<byte>((7 << 5) | 23));
			return m_stream.flush();
		}

		auto write(uint64_t x)
		{
			details::write_integer<0>(m_stream, x);
			return m_stream.flush();
		}
		auto write(int64_t x)
		{
			if (x < 0)
			{
				details::write_integer<1>(m_stream, static_cast<uint64_t>(-1ll - x));
				return m_stream.flush();
			}
			else
			{
				return write(static_cast<uint64_t>(x));
			}
		}

		auto start_binary(uint64_t cb)
		{
			details::write_integer<2>(m_stream, cb);
			return std::move(m_stream);
		}
		indefinite_stream_writer<Stream, 2> start_binary()
		{
			stream::write(m_stream, static_cast<byte>((2 << 5) | 31));
			return{ std::move(m_stream) };
		}

		auto start_string(uint64_t cb)
		{
			details::write_integer<3>(m_stream, cb);
			return std::move(m_stream);
		}
		indefinite_stream_writer<Stream, 3> start_string()
		{
			stream::write(m_stream, static_cast<byte>((3 << 5) | 31));
			return{ std::move(m_stream) };
		}

		array_writer<Stream> start_array(uint64_t size);
		indefinite_array_writer<Stream> start_array();

		map_writer<Stream> start_map(uint64_t size);
		indefinite_map_writer<Stream> start_map();
	private:
		Stream m_stream;
	};
	template <class Stream> document_writer<std::decay_t<Stream>> create_writer_no_debug_check(Stream&& s)
	{
		return{ std::forward<Stream>(s) };
	}
	template <class Stream, class error_handler> auto create_writer(Stream&& s, error_handler e)
	{
		return sax::make_writer(debug_checks::add_write_checks(create_writer_no_debug_check(std::forward<Stream>(s)), e));
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

		auto append() { return create_writer_no_debug_check(stream::ref(m_stream)); }
		auto flush() { return m_stream.flush(); }
	private:
		Stream m_stream;
	};
	template <class Stream> class indefinite_array_writer
	{
	public:
		indefinite_array_writer(Stream&& s)
			: m_stream(std::move(s))
		{}
		auto append() { return create_writer_no_debug_check(stream::ref(m_stream)); }
		auto flush()
		{
			stream::write(m_stream, static_cast<byte>(0xFF));
			return m_stream.flush();
		}
	private:
		Stream m_stream;
	};
	template <class Stream> array_writer<Stream> document_writer<Stream>::start_array(uint64_t size)
	{
		details::write_integer<4>(m_stream, size);
		return{ std::move(m_stream) };
	}
	template <class Stream> indefinite_array_writer<Stream> document_writer<Stream>::start_array()
	{
		stream::write(m_stream, static_cast<byte>((4 << 5) | 31));
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
		auto flush() { return m_stream.flush(); }
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
		auto flush()
		{
			stream::write(m_stream, static_cast<byte>(0xFF));
			return m_stream.flush();
		}
	private:
		Stream m_stream;
	};
	template <class Stream> map_writer<Stream> document_writer<Stream>::start_map(uint64_t size)
	{
		details::write_integer<5>(m_stream, size);
		return{ std::move(m_stream) };
	}
	template <class Stream> indefinite_map_writer<Stream> document_writer<Stream>::start_map()
	{
		stream::write(m_stream, static_cast<byte>((5 << 5) | 31));
		return{ std::move(m_stream) };
	}
}}