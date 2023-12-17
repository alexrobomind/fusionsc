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
	
	template<typename error_handler, class Document> using document_base = document_impl<
		Document::does_json_conversions,
		bool,
		std::nullptr_t,
		uint64_t,
		int64_t,
		double,
		undefined,
		string<error_handler, typename Document::template type_with_tag_t<tags::string>, tags::string>,
		string<error_handler, typename Document::template type_with_tag_t<tags::binary>, tags::binary>,
		array<error_handler, typename Document::template type_with_tag_t<tags::array>>,
		map<error_handler, typename Document::template type_with_tag_t<tags::map>>>;

	template <class error_handler, class Document> struct document : document_base<error_handler, Document>
	{
		using document_base<error_handler, Document>::document_base;
	};

	template <class error_handler, class Document> document<error_handler, std::decay_t<Document>> add_read_checks_impl(::goldfish::debug_checks::container_base<error_handler>* parent, Document&& t);

	template <class error_handler, class T, class _tag> class string : private ::goldfish::debug_checks::container_base<error_handler>
	{
	public:
		using tag = _tag;
		string(::goldfish::debug_checks::container_base<error_handler>* parent, T&& inner)
			: ::goldfish::debug_checks::container_base<error_handler>(parent)
			, m_inner(std::move(inner))
		{}

		size_t read_partial_buffer(buffer_ref buffer)
		{
			if (buffer.empty())
				return 0;

			auto result = m_inner.read_partial_buffer(buffer);
			if (result == 0)
				this -> unlock_parent();
			return result;
		}
		uint64_t seek(uint64_t cb)
		{
			auto skipped = stream::seek(m_inner, cb);
			if (skipped < cb)
				this -> unlock_parent();
			return skipped;
		}
	private:
		T m_inner;
	};
	template <class error_handler, class Tag, class T> string<error_handler, std::decay_t<T>, Tag> make_string(::goldfish::debug_checks::container_base<error_handler>* parent, T&& inner) { return{ parent, std::forward<T>(inner) }; }

	template <class error_handler, class T> class array : private ::goldfish::debug_checks::container_base<error_handler>
	{
	public:
		using tag = tags::array;

		array(::goldfish::debug_checks::container_base<error_handler>* parent, T&& inner)
			: ::goldfish::debug_checks::container_base<error_handler>(parent)
			, m_inner(std::move(inner))
		{}

		optional<decltype(add_read_checks_impl(static_cast<::goldfish::debug_checks::container_base<error_handler>*>(nullptr) /*parent*/, *std::declval<T>().read()))> read()
		{
			this -> err_if_locked();

			auto d = m_inner.read();
			if (d)
			{
				return add_read_checks_impl(this /*parent*/, std::move(*d));
			}
			else
			{
				this -> unlock_parent();
				return nullopt;
			}
		}
	private:
		T m_inner;
	};
	template <class error_handler, class T> array<error_handler, std::decay_t<T>> make_array(::goldfish::debug_checks::container_base<error_handler>* parent, T&& inner) { return{ parent, std::forward<T>(inner) }; }

	// The inheritance is public so that schema.h can use container_base to more aggressively lock the parent
	template <class error_handler, class T> class map : public ::goldfish::debug_checks::container_base<error_handler>
	{
	public:
		using tag = tags::map;

		map(::goldfish::debug_checks::container_base<error_handler>* parent, T&& inner)
			: ::goldfish::debug_checks::container_base<error_handler>(parent)
			, m_inner(std::move(inner))
		{}

		optional<decltype(add_read_checks_impl(static_cast<::goldfish::debug_checks::container_base<error_handler>*>(nullptr) /*parent*/, *std::declval<T>().read_key()))> read_key()
		{
			this -> err_if_locked();
			this -> err_if_flag_set();

			if (auto d = m_inner.read_key())
			{
				this -> set_flag();
				return add_read_checks_impl(this /*parent*/, std::move(*d));
			}
			else
			{
				this -> unlock_parent();
				return nullopt;
			}
		}
		auto read_value()
		{
			this -> err_if_locked();
			this -> err_if_flag_not_set();
			this -> clear_flag();
			return add_read_checks_impl(this /*parent*/, m_inner.read_value());
		}
	private:
		T m_inner;
	};
	template <class error_handler, class T> map<error_handler, std::decay_t<T>> make_map(::goldfish::debug_checks::container_base<error_handler>* parent, T&& inner) { return{ parent, std::forward<T>(inner) }; }

	template <class error_handler, class Document> document<error_handler, std::decay_t<Document>> add_read_checks_impl(::goldfish::debug_checks::container_base<error_handler>* parent, Document&& t)
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
			[&](auto&& x, tags::binary) -> document<error_handler, std::decay_t<Document>>
			{
				return make_string<error_handler, tags::binary>(parent, std::forward<decltype(x)>(x));
			},
			[&](auto&& x, tags::string) -> document<error_handler, std::decay_t<Document>>
			{
				return make_string<error_handler, tags::string>(parent, std::forward<decltype(x)>(x));
			},
			[](auto&& x, auto) -> document<error_handler, std::decay_t<Document>>
			{
				return std::forward<decltype(x)>(x);
			}));
	}

	template <class error_handler, class Document> auto add_read_checks(Document&& t, error_handler)
	{
		return add_read_checks_impl(static_cast<::goldfish::debug_checks::container_base<error_handler>*>(nullptr) /*parent*/, std::forward<Document>(t));
	}
	template <class Document> auto add_read_checks(Document&& t, ::goldfish::debug_checks::no_check)
	{
		return std::forward<Document>(t);
	}
}}
