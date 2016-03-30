#pragma once

#include <iostream>
#include "array_ref.h"
#include "stream.h"

namespace goldfish { namespace stream
{
	class istream_reader_ref
	{
	public:
		istream_reader_ref(std::istream& stream)
			: m_stream(stream)
		{}
		size_t read_buffer(buffer_ref buffer)
		{
			m_stream.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
			if (m_stream.bad() || (m_stream.fail() && !m_stream.eof()))
				throw io_exception{};

			return static_cast<size_t>(m_stream.gcount());
		}
	private:
		std::istream& m_stream;
	};
	class ostream_writer_ref
	{
	public:
		ostream_writer_ref(std::ostream& stream)
			: m_stream(stream)
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			m_stream.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
			if (m_stream.fail())
				throw io_exception{};
		}
		void flush()
		{
			m_stream.flush();
			if (m_stream.fail())
				throw io_exception{};
		}
	private:
		std::ostream& m_stream;
	};

	template <class inner, size_t N> class streambuf_on_reader : public std::streambuf
	{
	public:
		streambuf_on_reader(inner&& stream)
			: m_stream(std::move(stream))
		{
			set_cb(0);
		}
		streambuf_on_reader(streambuf_on_reader&& rhs)
			: m_stream(std::move(rhs.m_stream))
		{
			set_cb(rhs.get_cb());
			std::copy(rhs.gptr(), rhs.egptr(), make_unchecked_array_iterator(gptr()));
		}
		streambuf_on_reader(const streambuf_on_reader&) = delete;
		streambuf_on_reader& operator = (const streambuf_on_reader&) = delete;
		streambuf_on_reader& operator = (streambuf_on_reader&&) = delete;

	protected:
		int_type underflow() override
		{
			if (gptr() < egptr()) // buffer not exhausted
				return traits_type::to_int_type(*gptr());

			auto cb = m_stream.read_buffer({
				reinterpret_cast<byte*>(m_buffer.data() + 1 /*putback space*/),
				reinterpret_cast<byte*>(m_buffer.data() + m_buffer.size()) });
			if (cb == 0)
				return traits_type::eof();

			set_cb(cb);
			return traits_type::to_int_type(*gptr());
		}
	private:
		size_t get_cb() const
		{
			return std::distance(gptr(), egptr());
		}
		void set_cb(size_t cb)
		{
			setg(
				m_buffer.data(),
				m_buffer.data() + 1 /*putback space*/,
				m_buffer.data() + 1 /*putback space*/ + cb);
		}

		inner m_stream;
		std::array<char, N + 1 /*putback space*/> m_buffer;
	};
	template <size_t N, class inner> streambuf_on_reader<std::decay_t<inner>, N> make_streambuf(inner&& stream) { return{ std::forward<inner>(stream) }; }

	template <class StreamBuf> class istream_with_specific_streambuf : public std::istream
	{
	public:
		istream_with_specific_streambuf(StreamBuf&& streambuf)
			: std::istream(&m_streambuf)
			, m_streambuf(std::move(streambuf))
		{}
		istream_with_specific_streambuf(const istream_with_specific_streambuf&) = delete;
		istream_with_specific_streambuf(istream_with_specific_streambuf&& rhs)
			: std::istream(&m_streambuf)
			, m_streambuf(std::move(rhs.m_streambuf))
		{}
	private:
		StreamBuf m_streambuf;
	};
	template <class StreamBuf> istream_with_specific_streambuf<std::decay_t<StreamBuf>> make_istream_on_streambuf(StreamBuf&& streambuf)
	{
		return{ std::forward<StreamBuf>(streambuf) };
	}

	template <size_t N, class inner>
	auto make_istream(inner&& reader)
	{
		return make_istream_on_streambuf(make_streambuf<N>(std::forward<inner>(reader)));
	}
}}

template <class Stream> goldfish::stream::enable_if_reader_t<Stream, std::ostream&> operator << (std::ostream& s, Stream&& reader)
{
	goldfish::stream::copy(reader, goldfish::stream::ostream_writer_ref{ s });
	return s;
}