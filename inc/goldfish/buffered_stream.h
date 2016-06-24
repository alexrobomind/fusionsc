#pragma once

#include "stream.h"

namespace goldfish { namespace stream
{
	template <size_t N, class inner> class buffered_reader
	{
	public:
		buffered_reader(inner&& stream)
			: m_stream(std::move(stream))
		{}
		buffered_reader(buffered_reader&& rhs)
			: m_stream(std::move(rhs.m_stream))
		{
			m_buffered = buffer_ref(m_buffer_data.data(), rhs.m_buffered.size());
			copy(rhs.m_buffered, m_buffered);
		}
		buffered_reader& operator = (const buffered_reader&) = delete;

		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, T> read()
		{
			return read_helper<T>(std::integral_constant<size_t, alignof(T)>(), std::bool_constant<sizeof(T) <= N>());
		}
		template <class T> std::enable_if_t<std::is_standard_layout<T>::value && sizeof(T) <= N, optional<T>> peek()
		{
			return peek_helper<T>(std::integral_constant<size_t, alignof(T)>());
		}

		size_t read_partial_buffer(buffer_ref data)
		{
			if (data.empty())
				return 0;

			if (m_buffered.empty())
				fill_in_buffer();

			auto cb = std::min(m_buffered.size(), data.size());
			copy(m_buffered.remove_front(cb), buffer_ref{ data.begin(), cb });
			return cb;
		}

		uint64_t seek(uint64_t x)
		{
			if (x <= m_buffered.size())
			{
				m_buffered.remove_front(static_cast<size_t>(x));
				return x;
			}
			else
			{
				auto skipped = m_buffered.size() + stream::seek(m_stream, x - m_buffered.size());
				m_buffered.clear();
				return skipped;
			}
		}
	private:
		template <class T, size_t alignment> T read_helper(std::integral_constant<size_t, alignment>, std::bool_constant<false>)
		{
			T t;
			if (read_full_buffer(*this, { reinterpret_cast<byte*>(&t), sizeof(t) }) != sizeof(t))
				throw unexpected_end_of_stream();
			return t;
		}
		template <class T> T read_helper(std::integral_constant<size_t, 1> /*alignment*/, std::bool_constant<true> /*fits*/)
		{
			if (m_buffered.size() < sizeof(T))
				fill_in_buffer_ensure_size(sizeof(T));
			auto* data = m_buffered.data();
			m_buffered.remove_front(sizeof(T));
			return reinterpret_cast<T&>(*data);
		}
		template <class T, size_t alignment> T read_helper(std::integral_constant<size_t, alignment>, std::bool_constant<true> /*fits*/)
		{
			if (m_buffered.size() < sizeof(T))
				fill_in_buffer_ensure_size(sizeof(T));
			T t;
			memcpy(&t, m_buffered.data(), sizeof(t));
			m_buffered.remove_front(sizeof(t));
			return t;
		}
		template <class T> optional<T> peek_helper(std::integral_constant<size_t, 1>)
		{
			if (m_buffered.size() < sizeof(T) && !try_fill_in_buffer_ensure_size(sizeof(T)))
				return nullopt;
			return reinterpret_cast<T&>(*m_buffered.data());
		}
		template <class T, size_t alignment> optional<T> peek_helper(std::integral_constant<size_t, alignment>)
		{
			if (m_buffered.size() < sizeof(T) && !try_fill_in_buffer_ensure_size(sizeof(T)))
				return nullopt;
			T t;
			memcpy(&t, m_buffered.data(), sizeof(t));
			return t;
		}

		void fill_in_buffer()
		{
			assert(m_buffered.empty());
			m_buffered = { m_buffer_data.data(), m_stream.read_partial_buffer(m_buffer_data) };
		}
		bool try_fill_in_buffer_ensure_size(size_t s)
		{
			assert(s <= N);
			memmove(m_buffer_data.data(), m_buffered.data(), m_buffered.size());
			m_buffered = { m_buffer_data.data(), m_buffered.size() };

			while (m_buffered.size() < s)
			{
				auto cb = m_stream.read_partial_buffer({ m_buffered.end(), m_buffer_data.data() + N });
				if (cb == 0)
					return false;
				m_buffered = { m_buffered.begin(), m_buffered.end() + cb };
			}

			return true;
		}
		void fill_in_buffer_ensure_size(size_t s)
		{
			if (!try_fill_in_buffer_ensure_size(s))
				throw unexpected_end_of_stream();
		}

		inner m_stream;
		buffer_ref m_buffered;
		std::array<byte, N> m_buffer_data;
	};

	template <size_t N, class inner>
	class buffered_writer
	{
	public:
		buffered_writer(inner&& stream)
			: m_stream(std::move(stream))
			, m_begin_free_space(m_buffer_data.data())
		{}
		buffered_writer(buffered_writer&& rhs)
			: m_buffer_data(rhs.m_buffer_data)
			, m_begin_free_space(m_buffer_data.data() + std::distance(rhs.m_buffer_data.data(), rhs.m_begin_free_space))
			, m_stream(std::move(rhs.m_stream))
		{}
		buffered_writer& operator = (const buffered_writer&) = delete;

		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, void> write(const T& t)
		{
			write_static<sizeof(t)>(reinterpret_cast<const byte*>(&t), std::bool_constant<(sizeof(t) < N)>());
		}
		void write_buffer(const_buffer_ref data)
		{
			if (data.size() <= cb_free())
			{
				m_begin_free_space = std::copy(data.begin(), data.end(), m_begin_free_space);
				return;
			}

			if (m_begin_free_space != m_buffer_data.data()) // If not all of the buffer is free
			{
				auto cb = cb_free();
				m_begin_free_space = std::copy(data.begin(), data.begin() + cb, m_begin_free_space);
				data.remove_front(cb);
				if (data.empty())
					return;
				send_data();
			}
			assert(m_begin_free_space == m_buffer_data.data());

			if (data.size() >= m_buffer_data.size())
				m_stream.write_buffer(data);
			else
				m_begin_free_space = std::copy(data.begin(), data.end(), m_begin_free_space);
		}
		auto flush()
		{
			send_data();
			return m_stream.flush();
		}
	private:
		size_t cb_free() const { return m_buffer_data.data() + N - m_begin_free_space; }
		template <size_t cb> void write_static(const byte* t, std::false_type /*small*/) { write_buffer({ t, cb }); }
		template <size_t cb> void write_static(const byte* t, std::true_type /*small*/)
		{
			if (cb_free() < cb)
				send_data();
			m_begin_free_space = std::copy(t, t + cb, m_begin_free_space);
		}
		template <> void write_static<1>(const byte* t, std::true_type /*small*/)
		{
			if (m_begin_free_space == m_buffer_data.data() + N)
				send_data();
			*(m_begin_free_space++) = *t;
		}

		void send_data()
		{
			m_stream.write_buffer({
				m_buffer_data.data(),
				m_begin_free_space
			});
			m_begin_free_space = m_buffer_data.data();
		}
		std::array<byte, N> m_buffer_data;
		byte* m_begin_free_space;
		inner m_stream;
	};
	template <size_t N, class inner> enable_if_reader_t<inner, buffered_reader<N, std::decay_t<inner>>> buffer(inner&& stream) { return{ std::forward<inner>(stream) }; }
	template <size_t N, class inner> enable_if_writer_t<inner, buffered_writer<N, std::decay_t<inner>>> buffer(inner&& stream) { return{ std::forward<inner>(stream) }; }
}}