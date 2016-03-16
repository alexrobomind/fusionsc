#pragma once

#include "stream.h"

namespace goldfish { namespace stream
{
	template <class T, class U>
	static size_t copy_and_pop(array_ref<T>& from, array_ref<U>& to)
	{
		auto to_copy = std::min(from.size(), to.size());
		return copy(from.remove_front(to_copy), to.remove_front(to_copy));
	}

	template <size_t N, class inner>
	class buffered_reader
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

		size_t read_buffer(buffer_ref data)
		{
			auto read_already = copy_and_pop(m_buffered, data);
			if (data.empty())
			{
				return read_already;
			}
			else if (data.size() < N)
			{
				fill_in_buffer();
				return read_already + copy_and_pop(m_buffered, data);
			}
			else
			{
				return read_already + m_stream.read_buffer(data);
			}
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
			if (read_buffer({ reinterpret_cast<uint8_t*>(&t), sizeof(t) }) != sizeof(t))
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
			m_buffered = { m_buffer_data.data(), m_stream.read_buffer(m_buffer_data) };
		}
		bool try_fill_in_buffer_ensure_size(size_t s)
		{
			assert(s <= N);
			memmove(m_buffer_data.data(), m_buffered.data(), m_buffered.size());
			auto cb = m_stream.read_buffer({ m_buffer_data.data() + m_buffered.size(), m_buffer_data.data() + N });
			m_buffered = { m_buffer_data.data(), m_buffered.size() + cb };

			return m_buffered.size() >= s;
		}
		void fill_in_buffer_ensure_size(size_t s)
		{
			if (!try_fill_in_buffer_ensure_size(s))
				throw unexpected_end_of_stream();
		}

		inner m_stream;
		buffer_ref m_buffered;
		std::array<uint8_t, N> m_buffer_data;
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
			: assert_work_done(std::move(rhs))
			, m_buffer_data(rhs.m_buffer_data)
			, m_begin_free_space(m_buffer_data.data() + std::distance(rhs.m_buffer_data.data(), rhs.m_begin_free_space))
			, m_stream(std::move(rhs.m_stream))
		{}
		buffered_writer& operator = (const buffered_writer&) = delete;

		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, void> write(const T& t)
		{
			write_static<sizeof(t)>(reinterpret_cast<const uint8_t*>(&t), std::bool_constant<(sizeof(t) < N)>());
		}
		void write_buffer(const_buffer_ref data)
		{
			if (m_begin_free_space != m_buffer_data.data()) // If not all of the buffer is free
			{
				if (data.size() <= cb_free())
				{
					std::copy(data.begin(), data.end(), stdext::make_unchecked_array_iterator(m_begin_free_space));
					m_begin_free_space += data.size();
					return;
				}
				else
				{
					auto cb = cb_free();
					m_begin_free_space = std::copy(data.begin(), data.begin() + cb, m_begin_free_space);
					data.remove_front(cb);
				}
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
		template <size_t cb> void write_static(const uint8_t* t, std::false_type /*small*/) { write_buffer({ t, cb }); }
		template <size_t cb> void write_static(const uint8_t* t, std::true_type /*small*/)
		{
			if (cb_free() < cb)
				send_data();
			m_begin_free_space = std::copy(t, t + cb, m_begin_free_space);
		}
		template <> void write_static<1>(const uint8_t* t, std::true_type /*small*/)
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
		std::array<uint8_t, N> m_buffer_data;
		uint8_t* m_begin_free_space;
		inner m_stream;
	};
	template <size_t N, class inner> enable_if_reader_t<inner, buffered_reader<N, std::decay_t<inner>>> buffer(inner&& stream) { return{ std::forward<inner>(stream) }; }
	template <size_t N, class inner> enable_if_writer_t<inner, buffered_writer<N, std::decay_t<inner>>> buffer(inner&& stream) { return{ std::forward<inner>(stream) }; }
}}