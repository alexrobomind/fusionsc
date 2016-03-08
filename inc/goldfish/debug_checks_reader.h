#pragma once

#include "array_ref.h"
#include "sax_reader.h"
#include "tags.h"
#include "uncaught_exception.h"
#include <type_traits>

#ifndef NDEBUG
namespace goldfish { namespace debug_check
{
	template <class T, class _tag> class string;
	template <class T> class array;
	template <class T> class map;
	struct undefined { using tag = tags::undefined; };

	template <class Document> using document = document_on_variant<
		bool,
		nullptr_t,
		int64_t,
		uint64_t,
		double,
		undefined,
		string<std::decay_t<decltype(std::declval<Document>().as<tags::text_string>())>, tags::text_string>,
		string<std::decay_t<decltype(std::declval<Document>().as<tags::byte_string>())>, tags::byte_string>,
		array<std::decay_t<decltype(std::declval<Document>().as<tags::array>())>>,
		map<std::decay_t<decltype(std::declval<Document>().as<tags::map>())>>>;

	template <class Document> document<std::decay_t<Document>> add_read_checks(Document&& t);
	template <class T> document<T> add_read_checks(document<T>&& t) { return std::move(t); }
	template <class T> document<T> add_read_checks(const document<T>& t) = delete;

	template <class T, class tag> auto add_read_checks(T&& t, tag)
	{
		return std::forward<T>(t);
	}
	template <class T> auto add_read_checks(T&&, tags::undefined)
	{
		return undefined{};
	}

	template <class T, class _tag> class string : private assert_work_done
	{
	public:
		using tag = _tag;
		string(T inner)
			: m_inner(std::move(inner))
		{}

		size_t read_buffer(buffer_ref buffer)
		{
			auto result = m_inner.read_buffer(buffer);
			if (result < buffer.size())
				mark_work_done();
			return result;
		}
		uint64_t skip(uint64_t cb)
		{
			auto skipped = stream::skip(m_inner, cb);
			if (skipped < cb)
				mark_work_done();
			return skipped;
		}
	private:
		T m_inner;
	};
	template <class T> auto add_read_checks(T&& t, tags::byte_string)
	{
		return string<std::decay_t<T>, tags::byte_string>(std::forward<T>(t));
	}
	template <class T> auto add_read_checks(T&& t, tags::text_string)
	{
		return string<std::decay_t<T>, tags::text_string>(std::forward<T>(t));
	}

	template <class T> class array : private assert_work_done
	{
	public:
		using tag = tags::array;

		array(T&& inner)
			: m_inner(std::move(inner))
		{}

		optional<decltype(add_read_checks(*std::declval<T>().read()))> read()
		{
			assert(!is_work_done());
			auto d = m_inner.read();
			if (d)
			{
				return add_read_checks(std::move(*d));
			}
			else
			{
				mark_work_done();
				return nullopt;
			}
		}
	private:
		T m_inner;
	};
	template <class T> auto add_read_checks(T&& t, tags::array)
	{
		return array<T>(std::forward<T>(t));
	}

	template <class T> class map : private assert_work_done
	{
	public:
		using tag = tags::map;

		map(T&& inner)
			: m_inner(std::move(inner))
		{}

		optional<decltype(add_read_checks(*std::declval<T>().read_key()))> read_key()
		{
			assert(!is_work_done());
			assert(m_expect_read_key);
			m_expect_read_key = false;
			auto d = m_inner.read_key();
			if (d)
			{
				return add_read_checks(std::move(*d));
			}
			else
			{
				mark_work_done();
				return nullopt;
			}
		}
		auto read_value()
		{
			assert(!is_work_done());
			assert(!m_expect_read_key);
			m_expect_read_key = true;
			return add_read_checks(m_inner.read_value());
		}
	private:
		T m_inner;
		bool m_expect_read_key = true;
	};
	template <class T> auto add_read_checks(T&& t, tags::map)
	{
		return map<T>(std::forward<T>(t));
	}

	template <class Document> document<std::decay_t<Document>> add_read_checks(Document&& t)
	{
		return std::forward<Document>(t).visit([](auto&& x, auto tag) -> document<std::decay_t<Document>>
		{
			return{ add_read_checks(std::forward<decltype(x)>(x), tag) };
		});
	}
}}
#else
namespace goldfish { namespace debug_check
{
	template <class T> decltype(auto) add_read_checks(T&& t)
	{
		return std::forward<T>(t);
	}
}}
#endif