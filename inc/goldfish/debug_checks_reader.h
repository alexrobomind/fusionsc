#pragma once

#include "array_ref.h"
#include "debug_checks.h"
#include "sax_reader.h"
#include "tags.h"
#include <type_traits>

namespace goldfish { namespace debug_checks
{
	template <class error_handler, class T, class _tag> class string;
	template <class error_handler, class T> class array;
	template <class error_handler, class T> class map;
	struct undefined { using tag = tags::undefined; };

	template <class error_handler, class Document> using document = document_on_variant<
		Document::does_json_conversions,
		bool,
		nullptr_t,
		int64_t,
		uint64_t,
		double,
		undefined,
		string<error_handler, typename Document::template type_with_tag_t<tags::text_string>, tags::text_string>,
		string<error_handler, typename Document::template type_with_tag_t<tags::byte_string>, tags::byte_string>,
		array<error_handler, typename Document::template type_with_tag_t<tags::array>>,
		map<error_handler, typename Document::template type_with_tag_t<tags::map>>>;

	template <class error_handler, class Document> document<error_handler, std::decay_t<Document>> add_read_checks_impl(container_base<error_handler>* parent, Document&& t);

	template <class error_handler, class T, class _tag> class string : private container_base<error_handler>
	{
	public:
		using tag = _tag;
		string(container_base<error_handler>* parent, T&& inner)
			: container_base<error_handler>(parent)
			, m_inner(std::move(inner))
		{}

		size_t read_buffer(buffer_ref buffer)
		{
			auto result = m_inner.read_buffer(buffer);
			if (result < buffer.size())
				unlock_parent();
			return result;
		}
		uint64_t seek(uint64_t cb)
		{
			auto skipped = stream::seek(m_inner, cb);
			if (skipped < cb)
				unlock_parent();
			return skipped;
		}
	private:
		T m_inner;
	};
	template <class error_handler, class Tag, class T> string<error_handler, std::decay_t<T>, Tag> make_string(container_base<error_handler>* parent, T&& inner) { return{ parent, std::forward<T>(inner) }; }

	template <class error_handler, class T> class array : private container_base<error_handler>
	{
	public:
		using tag = tags::array;

		array(container_base<error_handler>* parent, T&& inner)
			: container_base<error_handler>(parent)
			, m_inner(std::move(inner))
		{}

		optional<decltype(add_read_checks_impl<error_handler>(nullptr /*parent*/, *std::declval<T>().read()))> read()
		{
			err_if_locked();

			auto d = m_inner.read();
			if (d)
			{
				return add_read_checks_impl<error_handler>(this /*parent*/, std::move(*d));
			}
			else
			{
				unlock_parent();
				return nullopt;
			}
		}
	private:
		T m_inner;
	};
	template <class error_handler, class T> array<error_handler, std::decay_t<T>> make_array(container_base<error_handler>* parent, T&& inner) { return{ parent, std::forward<T>(inner) }; }

	template <class error_handler, class T> class map : private container_base<error_handler>
	{
	public:
		using tag = tags::map;

		map(container_base<error_handler>* parent, T&& inner)
			: container_base<error_handler>(parent)
			, m_inner(std::move(inner))
		{}

		optional<decltype(add_read_checks_impl<error_handler>(nullptr /*parent*/, *std::declval<T>().read_key()))> read_key()
		{
			err_if_locked();
			err_if_flag_set();

			if (auto d = m_inner.read_key())
			{
				set_flag();
				return add_read_checks_impl<error_handler>(this /*parent*/, std::move(*d));
			}
			else
			{
				unlock_parent();
				return nullopt;
			}
		}
		auto read_value()
		{
			err_if_locked();
			err_if_flag_not_set();
			clear_flag();
			return add_read_checks_impl<error_handler>(this /*parent*/, m_inner.read_value());
		}
	private:
		T m_inner;
	};
	template <class error_handler, class T> map<error_handler, std::decay_t<T>> make_map(container_base<error_handler>* parent, T&& inner) { return{ parent, std::forward<T>(inner) }; }

	template <class error_handler, class Document> document<error_handler, std::decay_t<Document>> add_read_checks_impl(container_base<error_handler>* parent, Document&& t)
	{
		return std::forward<Document>(t).visit(first_match(
			[&](auto&& x, tags::map) -> document<error_handler, std::decay_t<Document>>
			{
				return make_map<error_handler>(parent, std::forward<decltype(x)>(x));
			},
			[&](auto&& x, tags::array) -> document<error_handler, std::decay_t<Document>>
			{
				return make_array<error_handler>(parent, std::forward<decltype(x)>(x));
			},
			[&](auto&& x, tags::byte_string) -> document<error_handler, std::decay_t<Document>>
			{
				return make_string<error_handler, tags::byte_string>(parent, std::forward<decltype(x)>(x));
			},
			[&](auto&& x, tags::text_string) -> document<error_handler, std::decay_t<Document>>
			{
				return make_string<error_handler, tags::text_string>(parent, std::forward<decltype(x)>(x));
			},
			[](auto&&, tags::undefined) -> document<error_handler, std::decay_t<Document>>
			{
				return undefined{};
			},
			[](auto&& x, auto) -> document<error_handler, std::decay_t<Document>>
			{
				return std::forward<decltype(x)>(x);
			}));
	}

	template <class error_handler, class Document> auto add_read_checks(Document&& t, error_handler)
	{
		return add_read_checks_impl<error_handler>(nullptr /*parent*/, std::forward<Document>(t));
	}
	template <class Document> auto add_read_checks(Document&& t, no_check)
	{
		return std::forward<Document>(t);
	}
}}
