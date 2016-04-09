#pragma once

#include <cstdint>
#include "common.h"
#include <type_traits>

namespace goldfish { namespace tags
{
	template <class T> struct is_tag : std::false_type {};
	struct binary {};         template <> struct is_tag<binary> : std::true_type {};
	struct string {};         template <> struct is_tag<string> : std::true_type {};
	struct array {};          template <> struct is_tag<array> : std::true_type {};
	struct map {};            template <> struct is_tag<map> : std::true_type {};
	struct undefined {};      template <> struct is_tag<undefined> : std::true_type {};
	struct floating_point {}; template <> struct is_tag<floating_point> : std::true_type {};
	struct unsigned_int {};   template <> struct is_tag<unsigned_int> : std::true_type {};
	struct signed_int {};     template <> struct is_tag<signed_int> : std::true_type {};
	struct boolean {};        template <> struct is_tag<boolean> : std::true_type {};
	struct null {};           template <> struct is_tag<null> : std::true_type {};
	struct document {};       template <> struct is_tag<document> : std::true_type {};
	using object = map;

	template <class T, class enable = void> struct tag { using type = typename T::tag; };
	template <> struct tag<uint64_t> { using type = unsigned_int; };
	template <> struct tag<int64_t> { using type = signed_int; };
	template <> struct tag<bool> { using type = boolean; };
	template <> struct tag<std::nullptr_t> { using type = null; };
	template <> struct tag<double> { using type = floating_point; };
	template <> struct tag<goldfish::undefined> { using type = undefined; };
	template <class T> using tag_t = typename tag<T>::type;

	template <class T> constexpr auto get_tag(T&&) { return tag_t<std::decay_t<T>>{}; }
	template <class T, class Tag> using has_tag = std::is_same<tag_t<T>, Tag>;

	template <class tag, class... T> struct contains_tag
	{
		enum { value = disjunction<has_tag<T, tag>...>::value };
	};

	template <class tag, class... T> struct type_with_tag {};
	template <class tag, bool pick_head, class... T> struct type_with_tag_helper {};
	template <class tag, class Head, class... Tail> struct type_with_tag_helper<tag, true, Head, Tail...>
	{
		static_assert(!contains_tag<tag, Tail...>::value, "Duplicate tag info");
		using type = Head; 
	};
	template <class tag, class Head, class... Tail> struct type_with_tag_helper<tag, false, Head, Tail...> { using type = typename type_with_tag<tag, Tail...>::type; };
	template <class tag, class Head, class... Tail> struct type_with_tag<tag, Head, Tail...>
	{
		using type = typename type_with_tag_helper<tag, has_tag<Head, tag>::value, Head, Tail...>::type;
	};
	template <class tag, class... T> using type_with_tag_t = typename type_with_tag<tag, T...>::type;
}}
