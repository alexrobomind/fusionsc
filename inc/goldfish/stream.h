#pragma once

#include <array>
#include <vector>
#include "array_ref.h"
#include "match.h"
#include "optional.h"

namespace goldfish { namespace stream 
{
	struct unexpected_end_of_stream : ill_formatted {};

	template <class T, class dummy = size_t> struct is_reader : std::false_type {};
	template <class T> struct is_reader<T, decltype(std::declval<T>().read_buffer(buffer_ref{})) > : std::true_type {};
	template <class T, class U> using enable_if_reader_t = std::enable_if_t<is_reader<std::decay_t<T>>::value, U>;

	template <class T, class dummy = void> struct is_writer : std::false_type {};
	template <class T> struct is_writer<T, decltype(std::declval<T>().write_buffer(const_buffer_ref{})) > : std::true_type {};
	template <class T, class U> using enable_if_writer_t = std::enable_if_t<is_writer<std::decay_t<T>>::value, U>;

	template <class T, class elem, class dummy = void> struct has_write : std::false_type {};
	template <class T, class elem> struct has_write<T, elem, decltype(std::declval<T>().write(std::declval<elem>()))> : std::true_type {};

	template <class T, class dummy = uint64_t> struct has_seek : std::false_type {};
	template <class T> struct has_seek<T, decltype(std::declval<T>().seek(0ull))> : std::true_type {};

	template <class T, class elem, class dummy = elem> struct has_read : std::false_type {};
	template <class T, class elem> struct has_read<T, elem, decltype(std::declval<T>().read<elem>())> : std::true_type {};

	template <class Stream> std::enable_if_t< has_seek<Stream>::value, uint64_t> seek(Stream& s, uint64_t x)
	{
		return s.seek(x);
	}
	template <class Stream> std::enable_if_t<!has_seek<Stream>::value, uint64_t> seek(Stream& s, uint64_t x)
	{
		auto original = x;
		byte buffer[8 * 1024];
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
		if (s.read_buffer({ reinterpret_cast<byte*>(&t), sizeof(t) }) != sizeof(t))
			throw unexpected_end_of_stream();
		return t;
	}

	template <class T, class stream> auto write(stream& s, const T& t) -> decltype(s.write(t)) { return s.write(t); }
	template <class T, class stream> enable_if_writer_t<stream, std::enable_if_t<std::is_standard_layout<T>::value && !has_write<stream, T>::value, void>> write(stream& s, const T& t)
	{
		s.write_buffer({ reinterpret_cast<const byte*>(&t), sizeof(t) });
	}

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
		uint64_t seek(uint64_t x) { return stream::seek(m_stream, x); }
		template <class T> auto peek() { return m_stream.peek<T>(); }
	private:
		inner& m_stream;
	};

	template <class inner> class ref_writer
	{
	public:
		static_assert(!is_ref<inner>::value, "Don't nest ref");

		ref_writer(inner& stream)
			: m_stream(stream)
		{}
		void write_buffer(const_buffer_ref data) { return m_stream.write_buffer(data); }
		template <class T> auto write(const T& t) { return stream::write(m_stream, t); }

		// Note that the ref_writer doesn't flush
		// The actual owner of the stream should be the one flushing
		void flush() { }
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

	class const_buffer_ref_reader
	{
	public:
		const_buffer_ref_reader() = default;
		const_buffer_ref_reader(const_buffer_ref_reader&&) = default;
		const_buffer_ref_reader& operator = (const const_buffer_ref_reader&) = delete;

		const_buffer_ref_reader(const_buffer_ref data)
			: m_data(data)
		{}
		size_t read_buffer(buffer_ref data)
		{
			auto to_copy = std::min(m_data.size(), data.size());
			return copy(m_data.remove_front(to_copy), data.remove_front(to_copy));
		}
		uint64_t seek(uint64_t x)
		{
			auto to_seek = static_cast<size_t>(std::min<uint64_t>(x, m_data.size()));
			m_data.remove_front(to_seek);
			return to_seek;
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

	inline const_buffer_ref_reader read_buffer_ref(const_buffer_ref x) { return{ x }; }
	template <size_t N> const_buffer_ref_reader read_string_ref(const char(&s)[N]) { assert(s[N - 1] == 0); return const_buffer_ref{ reinterpret_cast<const uint8_t*>(s), N - 1 }; }
	inline const_buffer_ref_reader read_string_ref(const char* s) { return const_buffer_ref{ reinterpret_cast<const uint8_t*>(s), strlen(s) }; }
	inline const_buffer_ref_reader read_string_ref(const std::string& s) { return const_buffer_ref{ reinterpret_cast<const uint8_t*>(s.data()), s.size() }; }

	class vector_reader : public const_buffer_ref_reader
	{
	public:
		vector_reader(std::vector<byte>&& buffer)
			: m_buffer(std::move(buffer))
		{
			m_data = m_buffer;
		}
		vector_reader(const vector_reader&) = delete;
		vector_reader(vector_reader&& rhs)
		{
			auto index_from = rhs.m_data.begin() - rhs.m_buffer.data();
			m_buffer = std::move(rhs.m_buffer);
			m_data = {
				m_buffer.data() + index_from,
				rhs.m_buffer.size()
			};
		}
		vector_reader& operator = (const vector_reader&) = delete;
		vector_reader& operator = (vector_reader&&) = delete;
	private:
		std::vector<byte> m_buffer;
	};
	class string_reader : public const_buffer_ref_reader
	{
	public:
		string_reader(std::string&& buffer)
			: m_buffer(std::move(buffer))
		{
			m_data = { reinterpret_cast<const byte*>(m_buffer.data()), m_buffer.size() };
		}
		string_reader(const string_reader&) = delete;
		string_reader(string_reader&& rhs)
		{
			auto index_from = rhs.m_data.begin() - reinterpret_cast<const byte*>(rhs.m_buffer.data());
			m_buffer = std::move(rhs.m_buffer);
			m_data = {
				reinterpret_cast<const byte*>(m_buffer.data()) + index_from,
				rhs.m_buffer.size()
			};
		}
		string_reader& operator = (const string_reader&) = delete;
		string_reader& operator = (string_reader&&) = delete;
	private:
		std::string m_buffer;
	};
	inline vector_reader read_buffer(std::vector<byte> x) { return{ std::move(x) }; }
	inline string_reader read_string(std::string x) { return{ std::move(x) }; }
	template <size_t N> auto read_string(const char(&s)[N]) { return read_string_ref(s); }

	class vector_writer
	{
	public:
		vector_writer() = default;
		vector_writer(vector_writer&&) = default;
		vector_writer& operator=(vector_writer&&) = default;

		void write_buffer(const_buffer_ref d)
		{
			m_data.insert(m_data.end(), d.begin(), d.end());
		}
		auto flush()
		{
			#ifndef NDEBUG
			m_flushed = true;
			#endif
			return std::move(m_data);
		}
		const auto& data()
		{
			assert(!m_flushed);
			return m_data;
		}
	private:
		#ifndef NDEBUG
		bool m_flushed = false;
		#endif
		std::vector<byte> m_data;
	};
	class string_writer
	{
	public:
		string_writer() = default;
		string_writer(string_writer&&) = default;
		string_writer& operator=(string_writer&&) = default;

		void write_buffer(const_buffer_ref d)
		{
			if (data.capacity() - data.size() < d.size())
				data.reserve(data.capacity() + data.capacity() / 2);

			data.append(reinterpret_cast<const char*>(d.begin()), reinterpret_cast<const char*>(d.end()));
		}
		template <class T> std::enable_if_t<std::is_standard_layout<T>::value && sizeof(T) == 1, void> write(const T& t)
		{
			data.push_back(reinterpret_cast<const char&>(t));
		}
		auto flush() { return std::move(data); }
	private:
		std::string data;
	};

	template <class stream> std::string read_all_as_string(stream&& s)
	{
		std::string result;
		byte buffer[65536];
		for (;;)
		{
			auto cb = s.read_buffer(buffer);
			result.insert(result.end(), reinterpret_cast<const char*>(buffer), reinterpret_cast<const char*>(buffer + cb));
			if (cb != sizeof(buffer))
				return result;
		}
	}

	template <class stream> enable_if_reader_t<stream, std::vector<byte>> read_all(stream&& s)
	{
		std::vector<byte> result;
		byte buffer[65536];
		while (auto cb = s.read_buffer(buffer))
			result.insert(result.end(), buffer, buffer + cb);
		return result;
	}
}}

template <class Stream> goldfish::stream::enable_if_reader_t<Stream, std::ostream&> operator << (std::ostream& s, Stream&& reader)
{
	goldfish::byte buffer[65536];
	while (auto cb = reader.read_buffer(buffer))
		s.write(reinterpret_cast<const char*>(buffer), cb);
	return s;
}