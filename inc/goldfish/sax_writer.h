#pragma once

#include "match.h"
#include "stream.h"
#include "tags.h"
#include <type_traits>

namespace goldfish { namespace sax
{
	template <class inner> class document_writer;
	template <class inner> document_writer<std::decay_t<inner>> make_writer(inner&& writer);

	template <class inner> class array_writer
	{
	public:
		array_writer(inner&& writer)
			: m_writer(std::move(writer))
		{}

		template <class T> void write(T&& t) { append().write(std::forward<T>(t)); }
		auto start_binary(uint64_t cb) { return append().start_binary(cb); }
		auto start_binary() { return append().start_binary(); }
		auto start_string(uint64_t cb) { return append().start_string(cb); }
		auto start_string() { return append().start_string(); }
		auto start_array(uint64_t size) { return append().start_array(size); }
		auto start_array() { return append().start_array(); }
		auto start_map(uint64_t size) { return append().start_map(size); }
		auto start_map() { return append().start_map(); }

		auto append() { return make_writer(m_writer.append()); }
		auto flush() { return m_writer.flush(); }
	private:
		inner m_writer;
	};
	template <class inner> array_writer<std::decay_t<inner>> make_array_writer(inner&& writer) { return{ std::forward<inner>(writer) }; }

	template <class inner> class map_writer
	{
	public:
		map_writer(inner&& writer)
			: m_writer(std::move(writer))
		{}

		auto append_key() { return make_writer(m_writer.append_key()); }
		auto append_value() { return make_writer(m_writer.append_value()); }

		// Write the key, don't start the value
		template <class T> void write_key(T&& t) { append_key().write(std::forward<T>(t)); }
		auto start_binary_key(uint64_t cb) { return append_key().start_binary(cb); }
		auto start_binary_key() { return append_key().start_binary(); }
		auto start_string_key(uint64_t cb) { return append_key().start_string(cb); }
		auto start_string_key() { return append_key().start_string(); }
		auto start_array_key(uint64_t size) { return append_key().start_array(size); }
		auto start_array_key() { return append_key().start_array(); }
		auto start_map_key(uint64_t size) { return append_key().start_map(size); }
		auto start_map_key() { return append_key().start_map(); }

		// Write the value after having used one of the above
		template <class T> void write_value(T&& t) { append_value().write(std::forward<T>(t)); }
		auto start_binary_value(uint64_t cb) { return append_value().start_binary(cb); }
		auto start_binary_value() { return append_value().start_binary(); }
		auto start_string_value(uint64_t cb) { return append_value().start_string(cb); }
		auto start_string_value() { return append_value().start_string(); }
		auto start_array_value(uint64_t size) { return append_value().start_array(size); }
		auto start_array_value() { return append_value().start_array(); }
		auto start_map_value(uint64_t size) { return append_value().start_map(size); }
		auto start_map_value() { return append_value().start_map(); }

		// Write the key, start the value
		template <class T> auto append(T&& t)
		{
			append_key().write(std::forward<T>(t));
			return append_value();
		}
		template <class T> auto start_binary(T&& key, uint64_t cb) { write_key(std::forward<T>(key)); return start_binary_value(cb); }
		template <class T> auto start_binary(T&& key) { write_key(std::forward<T>(key)); return start_binary_value(); }
		template <class T> auto start_string(T&& key, uint64_t cb) { write_key(std::forward<T>(key)); return start_string_value(cb); }
		template <class T> auto start_string(T&& key) { write_key(std::forward<T>(key)); return start_string_value(); }
		template <class T> auto start_array(T&& key, uint64_t size) { write_key(std::forward<T>(key)); return start_array_value(size); }
		template <class T> auto start_array(T&& key) { write_key(std::forward<T>(key)); return start_array_value(); }
		template <class T> auto start_map(T&& key, uint64_t size) { write_key(std::forward<T>(key)); return start_map_value(size); }
		template <class T> auto start_map(T&& key) { write_key(std::forward<T>(key)); return start_map_value(); }

		// Write both the key and the value
		template <class T, class U> void write(T&& key, U&& value)
		{
			write_key(std::forward<T>(key));
			write_value(std::forward<U>(value));
		}

		auto flush() { return m_writer.flush(); }
	private:
		inner m_writer;
	};
	template <class inner> map_writer<std::decay_t<inner>> make_map_writer(inner&& writer) { return{ std::forward<inner>(writer) }; }

	template <class inner> class document_writer
	{
	public:
		document_writer(inner&& writer)
			: m_writer(std::move(writer))
		{}

		auto write(bool x)       { return m_writer.write(x); }
		auto write(nullptr_t x)  { return m_writer.write(x); }
		auto write(double x)     { return m_writer.write(x); }
		auto write(undefined x)  { return m_writer.write(x); }

		auto write(uint64_t x)   { return m_writer.write(x); }
		auto write(uint32_t x)   { return m_writer.write(static_cast<uint64_t>(x)); }
		auto write(uint16_t x)   { return m_writer.write(static_cast<uint64_t>(x)); }
		auto write(uint8_t x)    { return m_writer.write(static_cast<uint64_t>(x)); }

		auto write(int64_t x)    { return m_writer.write(x); }
		auto write(int32_t x)    { return m_writer.write(static_cast<int64_t>(x)); }
		auto write(int16_t x)    { return m_writer.write(static_cast<int64_t>(x)); }
		auto write(int8_t x)     { return m_writer.write(static_cast<int64_t>(x)); }

		auto start_binary(uint64_t cb) { return m_writer.start_binary(cb); }
		auto start_binary() { return m_writer.start_binary(); }
		auto write(const_buffer_ref x)
		{
			auto stream = start_binary(x.size());
			stream.write_buffer(x);
			return stream.flush();
		}

		auto start_string(uint64_t cb) { return m_writer.start_string(cb); }
		auto start_string() { return m_writer.start_string(); }
		auto write(const char* text) { return write(text, strlen(text)); }
		auto write(const std::string& text) { return write(text.data(), text.size()); }
		template <size_t N> auto write(const char(&text)[N])
		{
			static_assert(N > 0, "Expect null terminated strings");
			assert(text[N - 1] == 0);
			return write(text, N - 1);
		}

		auto start_array(uint64_t size) { return make_array_writer(m_writer.start_array(size)); }
		auto start_array() { return make_array_writer(m_writer.start_array()); }

		auto start_map(uint64_t size) { return make_map_writer(m_writer.start_map(size)); }
		auto start_map() { return make_map_writer(m_writer.start_map()); }

		template <class T> auto write(T&& s, std::enable_if_t<stream::is_reader<std::decay_t<T>>::value>* = nullptr)
		{
			return copy(s, [&](size_t cb) { return start_binary(cb); }, [&] { return start_binary(); });
		}

		template <class T> auto write(T&& document, std::enable_if_t<std::is_same<typename std::decay_t<T>::tag, tags::document>::value>* = nullptr)
		{
			return document.visit(best_match(
				[&](auto&& x, tags::binary) { return write(x); },
				[&](auto&& x, tags::string) { return copy(x, [&](size_t cb) { return start_string(cb); }, [&] { return start_string(); }); },
				[&](auto&& x, tags::array)
				{
					auto array_writer = start_array();
					while (auto element = x.read())
						array_writer.write(*element);
					return array_writer.flush();
				},
				[&](auto&& x, tags::map)
				{
					auto map_writer = start_map();
					while (auto key = x.read_key())
					{
						map_writer.write_key(*key);
						map_writer.write_value(x.read_value());
					}
					return map_writer.flush();
				},
				[&](auto&& x, tags::undefined) { return write(x); },
				[&](auto&& x, tags::floating_point) { return write(x); },
				[&](auto&& x, tags::unsigned_int) { return write(x); },
				[&](auto&& x, tags::signed_int) { return write(x); },
				[&](auto&& x, tags::boolean) { return write(x); },
				[&](auto&& x, tags::null) { return write(x); }
			));
		}

		template <class T> decltype(serialize_to_goldfish(std::declval<document_writer<inner>&>(), std::declval<T&&>())) write(T&& t)
		{
			return serialize_to_goldfish(*this, std::forward<T>(t));
		}
	private:
		template <class Stream, class CreateWriterWithSize, class CreateWriterWithoutSize>
		auto copy(Stream& s, CreateWriterWithSize&& create_writer_with_size, CreateWriterWithoutSize&& create_writer_without_size)
		{
			byte buffer[typical_buffer_length];
			auto cb = stream::read_full_buffer(s, buffer);
			if (cb < sizeof(buffer))
			{
				// We read the entire stream
				auto output_stream = create_writer_with_size(cb);
				output_stream.write_buffer({ buffer, cb });
				return output_stream.flush();
			}
			else
			{
				// We read only a portion of the stream
				auto output_stream = create_writer_without_size();
				output_stream.write_buffer(buffer);
				stream::copy(s, output_stream);
				return output_stream.flush();
			}
		}

		auto write(const char* text, size_t length)
		{
			auto stream = start_string(length);
			stream.write_buffer({ reinterpret_cast<const byte*>(text), length });
			return stream.flush();
		}
		inner m_writer;
	};

	template <class inner> document_writer<std::decay_t<inner>> make_writer(inner&& writer) { return{ std::forward<inner>(writer) }; }
}}
