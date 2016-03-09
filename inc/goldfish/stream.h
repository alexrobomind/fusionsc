#pragma once

#include <array>
#include <vector>
#include "array_ref.h"
#include "uncaught_exception.h"
#include "match.h"
#include "optional.h"

namespace goldfish { namespace stream 
{
	struct unexpected_end_of_stream {};

	template <class T, class dummy = size_t> struct is_reader : std::false_type {};
	template <class T> struct is_reader<T, decltype(std::declval<T>().read_buffer(buffer_ref{})) > : std::true_type {};
	template <class T, class U> using enable_if_reader_t = std::enable_if_t<is_reader<std::decay_t<T>>::value, U>;

	template <class T, class dummy = void> struct is_writer : std::false_type {};
	template <class T> struct is_writer<T, decltype(std::declval<T>().write_buffer(const_buffer_ref{})) > : std::true_type {};
	template <class T, class U> using enable_if_writer_t = std::enable_if_t<is_writer<std::decay_t<T>>::value, U>;

	template <class T, class elem, class dummy = void> struct has_write : std::false_type {};
	template <class T, class elem> struct has_write<T, elem, decltype(std::declval<T>().write(std::declval<elem>()))> : std::true_type {};

	template <class T, class dummy = uint64_t> struct has_skip : std::false_type {};
	template <class T> struct has_skip<T, decltype(std::declval<T>().skip(0ull))> : std::true_type {};

	template <class T, class elem, class dummy = elem> struct has_read : std::false_type {};
	template <class T, class elem> struct has_read<T, elem, decltype(std::declval<T>().read<elem>())> : std::true_type {};

	template <class Stream> std::enable_if_t< has_skip<Stream>::value, uint64_t> skip(Stream& s, uint64_t x)
	{
		return s.skip(x);
	}
	template <class Stream> std::enable_if_t<!has_skip<Stream>::value, uint64_t> skip(Stream& s, uint64_t x)
	{
		auto original = x;
		uint8_t buffer[8 * 1024];
		while (x >= sizeof(buffer))
		{
			auto cb = s.read_buffer(buffer);
			x -= cb;
			if (cb < sizeof(buffer))
				return original - x;
		}
		x -= s.read_buffer({ buffer, buffer + x });
		return original - x;
	}

	template <class T, class Stream> std::enable_if_t< has_read<Stream, T>::value, T> read(Stream& s)
	{
		return s.read<T>();
	}

	template <class T, class Stream> enable_if_reader_t<Stream, std::enable_if_t<!has_read<Stream, T>::value, T>> read(Stream& s)
	{
		T t;
		if (s.read_buffer({ reinterpret_cast<uint8_t*>(&t), sizeof(t) }) != sizeof(t))
			throw unexpected_end_of_stream();
		return t;
	}

	template <class T, class stream> auto write(stream& s, const T& t) -> decltype(s.write(t)) { return s.write(t); }
	template <class T, class stream> enable_if_writer_t<stream, std::enable_if_t<std::is_standard_layout<T>::value && !has_write<stream, T>::value, void>> write(stream& s, const T& t)
	{
		s.write_buffer({ reinterpret_cast<const uint8_t*>(&t), sizeof(t) });
	}

	class empty
	{
	public:
		size_t read_buffer(buffer_ref) { return 0; }
		uint64_t skip(uint64_t) { return 0; }
		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, optional<T>> peek() { return nullopt; }
	};

	template <class inner> class ref_reader;
	template <class inner> class ref_writer;
	template <class T> struct is_ref : std::false_type {};
	template <class T> struct is_ref<ref_reader<T>> : std::true_type {};
	template <class T> struct is_ref<ref_writer<T>> : std::true_type {};

	template <class inner>
	class ref_reader
	{
	public:
		static_assert(!is_ref<inner>::value, "Don't nest ref");

		ref_reader(inner& stream)
			: m_stream(stream)
		{}
		size_t read_buffer(buffer_ref data) { return m_stream.read_buffer(data); }
		template <class T> auto read() { return stream::read<T>(m_stream); }
		uint64_t skip(uint64_t x) { return stream::skip(m_stream, x); }
		template <class T> auto peek() { return m_stream.peek<T>(); }
	private:
		inner& m_stream;
	};

	template <class inner>
	class ref_writer
	{
	public:
		static_assert(!is_ref<inner>::value, "Don't nest ref");

		ref_writer(inner& stream)
			: m_stream(stream)
		{}
		void write_buffer(const_buffer_ref data) { return m_stream.write_buffer(data); }
		template <class T> auto write(const T& t) { return stream::write(m_stream, t); }
		void flush() { m_stream.flush(); }
	private:
		inner& m_stream;
	};

	template <class inner> struct reader_ref_type { using type = ref_reader<inner>; };
	template <class inner> struct reader_ref_type<ref_reader<inner>> { using type = ref_reader<inner>; };
	template <class inner> using reader_ref_type_t = typename reader_ref_type<inner>::type;

	template <class inner> struct writer_ref_type { using type = ref_writer<inner>; };
	template <class inner> struct writer_ref_type<ref_writer<inner>> { using type = ref_writer<inner>; };
	template <class inner> using writer_ref_type_t = typename writer_ref_type<inner>::type;

	template <class inner> std::enable_if_t<is_reader<inner>::value && !is_ref<inner>::value, ref_reader<inner>> ref(inner& stream) { return{ stream }; }
	template <class inner> std::enable_if_t<is_writer<inner>::value && !is_ref<inner>::value, ref_writer<inner>> ref(inner& stream) { return{ stream }; }
	template <class inner> ref_reader<inner> ref(ref_reader<inner>& stream) { return stream; }
	template <class inner> ref_writer<inner> ref(ref_writer<inner>& stream) { return stream; }

	template <class inner>
	class assert_read_entirely_reader
	{
	public:
		assert_read_entirely_reader(inner&& stream)
			: m_stream(std::move(stream))
		{}
		assert_read_entirely_reader(assert_read_entirely_reader&&) = default;
		~assert_read_entirely_reader()
		{
			assert(!m_work_needed() || m_uncaught_exceptions() || stream::skip(m_stream, 1) == 0);
		}
		size_t read_buffer(buffer_ref data) { return m_stream.read_buffer(data); }
		template <class T> auto read() { return stream::read<T>(m_stream); }
		uint64_t skip(uint64_t x) { return stream::skip(m_stream, x); }
	private:
		inner m_stream;
		uncaught_exception_checker m_uncaught_exceptions;
		is_work_needed m_work_needed;
	};
	#if _DEBUG
	template <class inner> enable_if_reader_t<inner, assert_read_entirely_reader<std::decay_t<inner>>> assert_read_entirely(inner&& stream) { return{ std::forward<inner>(stream) }; }
	#else
	template <class inner> enable_if_reader_t<inner, inner> assert_read_entirely(inner stream) { return std::move(stream); }
	#endif


	class array_ref_reader
	{
	public:
		array_ref_reader() = default;
		array_ref_reader(array_ref_reader&&) = default;
		array_ref_reader& operator = (const array_ref_reader&) = delete;

		array_ref_reader(const_buffer_ref data)
			: m_data(data)
		{}
		size_t read_buffer(buffer_ref data)
		{
			return copy_and_pop(m_data, data);
		}
		uint64_t skip(uint64_t x)
		{
			auto to_skip = static_cast<size_t>(std::min<uint64_t>(x, m_data.size()));
			m_data.remove_front(to_skip);
			return to_skip;
		}

		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, T> read()
		{
			return read_helper<T>(std::integral_constant<size_t, sizeof(T)>());
		}
		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, optional<T>> peek()
		{
			return peek_helper<T>(std::integral_constant<size_t, sizeof(T)>());
		}
	private:
		template <class T> optional<T> peek_helper(std::integral_constant<size_t, 1>)
		{
			if (m_data.empty())
				return nullopt;
			return reinterpret_cast<const T&>(m_data.front());
		}
		template <class T, size_t s> optional<T> peek_helper(std::integral_constant<size_t, s>)
		{
			if (m_data.size() < sizeof(T))
				return nullopt;
			T t;
			memcpy(&t, m_data.data(), sizeof(t));
			return t;
		}
		template <class T> T read_helper(std::integral_constant<size_t, 1>)
		{
			if (m_data.empty())
				throw unexpected_end_of_stream();
			return reinterpret_cast<const T&>(m_data.pop_front());
		}
		template <class T, size_t s> T read_helper(std::integral_constant<size_t, s>)
		{
			if (m_data.size() < sizeof(T))
				throw unexpected_end_of_stream();
			T t;
			memcpy(&t, m_data.data(), sizeof(t));
			m_data.remove_front(sizeof(t));
			return t;
		}

		const_buffer_ref data() const { return m_data; }
	protected:
		const_buffer_ref m_data;
	};

	template <size_t N>
	class array_reader : public array_ref_reader
	{
	public:
		array_reader(const std::array<uint8_t, N>& buffer)
			: array_ref_reader(m_buffer)
			, m_buffer(buffer)
		{}
		
		template <size_t M, std::enable_if_t<M < N, bool> = true>
		array_reader(const std::array<uint8_t, M>& buffer)
		{
			std::copy(buffer.begin(), buffer.end(), m_buffer.begin());
			m_data = { m_buffer.data(), m_buffer.data() + M };
		}
		array_reader(const array_reader& rhs)
			: m_buffer(rhs.m_buffer)
		{
			m_data = { m_buffer.data() + std::distance(rhs.m_buffer.data(), rhs.m_data.data()), rhs.m_data.size() };
		}

	private:
		std::array<uint8_t, N> m_buffer;
	};
	class vector_reader : public array_ref_reader
	{
	public:
		vector_reader(std::vector<uint8_t> buffer)
			: m_buffer(std::move(buffer))
		{
			m_data = { m_buffer.data(), m_buffer.data() + m_buffer.size() };
		}
		vector_reader(vector_reader&& rhs)
		{
			auto d = std::distance(const_cast<const uint8_t*>(rhs.m_buffer.data()), rhs.m_data.data());
			auto s = rhs.m_data.size();

			m_buffer = std::move(rhs.m_buffer);
			m_data = { m_buffer.data() + d, s };
		}

	private:
		std::vector<uint8_t> m_buffer;
	};
	template <size_t N> array_reader<N> read_array(const std::array<uint8_t, N>& x) { return{ x }; }
	inline vector_reader read_array(std::vector<uint8_t> x) { return{ std::move(x) }; }
	inline array_ref_reader read_array_ref(const_buffer_ref x) { return{ x }; }
	template <size_t N> array_ref_reader read_string_literal(const char(&s)[N]) { assert(s[N - 1] == 0); return const_buffer_ref{ reinterpret_cast<const uint8_t*>(s), N - 1 }; }
	inline array_ref_reader read_string_literal(const char* s) { return const_buffer_ref{ reinterpret_cast<const uint8_t*>(s), strlen(s) }; }

	struct vector_writer : private assert_work_done
	{
		vector_writer() = default;
		vector_writer(vector_writer&&) = default;
		vector_writer& operator=(vector_writer&&) = default;

		void write_buffer(const_buffer_ref d)
		{
			data.insert(data.end(), d.begin(), d.end());
		}
		void flush()
		{
			mark_work_done();
		}
		std::vector<uint8_t> data;
	};
	struct string_writer : private assert_work_done
	{
		string_writer() = default;
		string_writer(string_writer&&) = default;
		string_writer& operator=(string_writer&&) = default;

		void write_buffer(const_buffer_ref d)
		{
			data.insert(data.end(), reinterpret_cast<const char*>(d.begin()), reinterpret_cast<const char*>(d.end()));
		}
		void flush()
		{
			mark_work_done();
		}
		std::string data;
	};

	template <size_t N, class inner>
	class buffered_reader
	{
	public:
		buffered_reader(inner&& stream)
			: m_stream(std::move(stream))
		{}
		buffered_reader(buffered_reader&& rhs)
			: m_buffer_data(rhs.m_buffer_data)
			, m_stream(std::move(rhs.m_stream))
		{
			if (!rhs.m_buffered.empty())
			{
				m_buffered = buffer_ref(
					m_buffer_data.data() + std::distance(rhs.m_buffer_data.data(), rhs.m_buffered.begin()),
					rhs.m_buffered.size());
			}
		}
		buffered_reader& operator = (const buffered_reader&) = delete;

		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, T> read()
		{
			return read_helper<T>(std::integral_constant<size_t, sizeof(T)>(), std::bool_constant<sizeof(T) <= N>());
		}
		template <class T> std::enable_if_t<std::is_standard_layout<T>::value && sizeof(T) <= N, optional<T>> peek()
		{
			return peek_helper<T>(std::integral_constant<size_t, sizeof(T)>());
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

		uint64_t skip(uint64_t x)
		{
			if (x <= m_buffered.size())
			{
				m_buffered.remove_front(static_cast<size_t>(x));
				return x;
			}
			else
			{
				auto skipped = m_buffered.size() + stream::skip(m_stream, x - m_buffered.size());
				m_buffered.clear();
				return skipped;
			}
		}
	private:
		template <class T, size_t s, bool fits> T read_helper(std::integral_constant<size_t, s>, std::bool_constant<fits>)
		{
			T t;
			if (self().read_buffer({ reinterpret_cast<uint8_t*>(&t), sizeof(t) }) != sizeof(t))
				throw unexpected_end_of_stream();
			return t;
		}
		template <class T, size_t s> T read_helper(std::integral_constant<size_t, 1>, std::bool_constant<true> /*fits*/)
		{
			if (m_buffered.empty())
				fill_in_buffer_ensure_size(1);
			return reinterpret_cast<T&>(m_buffered.pop_front());
		}
		template <class T, size_t s> T read_helper(std::integral_constant<size_t, s>, std::bool_constant<true> /*fits*/)
		{
			if (m_buffered.size() < s)
				fill_in_buffer_ensure_size(s);
			T t;
			memcpy(&t, m_buffered.data(), sizeof(t));
			m_buffered.remove_front(sizeof(t));
			return t;
		}
		template <class T> optional<T> peek_helper(std::integral_constant<size_t, 1>)
		{
			if (m_buffered.empty() && !try_fill_in_buffer_ensure_size(1))
				return nullopt;
			return reinterpret_cast<T&>(m_buffered.front());
		}
		template <class T, size_t s> optional<T> peek_helper(std::integral_constant<size_t, s>)
		{
			if (m_buffered.size() < s && !try_fill_in_buffer_ensure_size(s))
				return nullopt;
			T t;
			memcpy(&t, m_buffered.data(), sizeof(t));
			return t;
		}

		void fill_in_buffer()
		{
			m_buffered = { m_buffer_data.data(), m_stream.read_buffer(m_buffer_data) };
		}
		bool try_fill_in_buffer_ensure_size(size_t s)
		{
			if (m_buffered.empty())
			{
				m_buffered = { m_buffer_data.data(), m_stream.read_buffer(m_buffer_data) };
			}
			else
			{
				memmove(m_buffer_data.data(), m_buffered.data(), m_buffered.size());
				auto cb = m_stream.read_buffer({ m_buffer_data.data() + m_buffered.size(), m_buffer_data.data() + N });
				m_buffered = { m_buffer_data.data(), m_buffered.size() + cb };
			}

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
	class buffered_writer : private assert_work_done
	{
	public:
		buffered_writer(inner&& stream)
			: m_stream(std::move(stream))
			, m_free_space(m_buffer_data)
		{}
		buffered_writer(buffered_writer&& rhs)
			: assert_work_done(std::move(rhs))
			, m_buffer_data(rhs.m_buffer_data)
			, m_free_space(
				m_buffer_data.data() + std::distance(rhs.m_buffer_data.data(), rhs.m_free_space.begin()),
				m_buffer_data.data() + std::distance(rhs.m_buffer_data.data(), rhs.m_free_space.end()))
			, m_stream(std::move(rhs.m_stream))
		{}
		buffered_writer& operator = (const buffered_writer&) = delete;

		template <class T> std::enable_if_t<std::is_standard_layout<T>::value, void> write(const T& t)
		{
			write_static<sizeof(t)>(reinterpret_cast<const uint8_t*>(&t), std::integral_constant<bool, (sizeof(t) < N / 2)>());
		}
		void write_buffer(const_buffer_ref data)
		{
			if (m_free_space.size() != m_buffer_data.size())
			{
				if (data.size() <= m_free_space.size())
				{
					std::copy(data.begin(), data.end(), stdext::make_unchecked_array_iterator(m_free_space.begin()));
					m_free_space.remove_front(data.size());
					return;
				}
				else
				{
					copy_and_pop(data, m_free_space);
				}
				if (data.empty())
					return;
				send_data();
			}
			assert(m_free_space.size() == m_buffer_data.size());

			if (data.size() >= m_buffer_data.size())
				m_stream.write_buffer(data);
			else
				copy_and_pop(data, m_free_space);
		}
		void flush()
		{
			m_stream.write_buffer({
				m_buffer_data.data(),
				m_free_space.data()
			});
			m_free_space = m_buffer_data;
			m_stream.flush();
			mark_work_done();
		}
	private:
		template <size_t cb> void write_static(const uint8_t* t, std::false_type /*small*/) { write_buffer({ t, cb }); }
		template <size_t cb> void write_static(const uint8_t* t, std::true_type /*small*/)
		{
			if (m_free_space.size() < cb)
				send_data();
			memcpy(m_free_space.data(), t, cb);
			m_free_space.remove_front(cb);
		}
		template <> void write_static<1>(const uint8_t* t, std::true_type /*small*/)
		{
			if (m_free_space.empty())
				send_data();
			m_free_space.pop_front() = *t;
		}

		void send_data()
		{
			m_stream.write_buffer({
				m_buffer_data.data(),
				m_free_space.data()
			});
			m_free_space = m_buffer_data;
		}
		std::array<uint8_t, N> m_buffer_data;
		buffer_ref m_free_space;
		inner m_stream;
	};
	template <size_t N, class inner> enable_if_reader_t<inner, buffered_reader<N, std::decay_t<inner>>> buffer(inner&& stream) { return{ std::forward<inner>(stream) }; }
	template <size_t N, class inner> enable_if_writer_t<inner, buffered_writer<N, std::decay_t<inner>>> buffer(inner&& stream) { return{ std::forward<inner>(stream) }; }

	template <class stream> std::string read_all_as_string(stream&& s)
	{
		std::string result;
		uint8_t buffer[65536];
		while (auto cb = s.read_buffer(buffer))
			result.insert(result.end(), reinterpret_cast<const char*>(buffer), reinterpret_cast<const char*>(buffer + cb));
		return result;
	}

	template <class stream> enable_if_reader_t<stream, std::vector<uint8_t>> read_all(stream&& s)
	{
		std::vector<uint8_t> result;
		uint8_t buffer[65536];
		while (auto cb = s.read_buffer(buffer))
			result.insert(result.end(), buffer, buffer + cb);
		return result;
	}
}}