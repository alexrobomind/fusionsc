#pragma once

#include <cstdint>
#include <type_traits>

namespace goldfish { namespace tags
{
	template <class T> struct is_tag : std::false_type {};

	struct byte_string {};       template <> struct is_tag<byte_string> : std::true_type {};
	struct text_string {};		 template <> struct is_tag<text_string> : std::true_type {};
	struct array {};			 template <> struct is_tag<array> : std::true_type {};
	struct map {};				 template <> struct is_tag<map> : std::true_type {};
	struct undefined {};		 template <> struct is_tag<undefined> : std::true_type {};
	struct floating_point {};	 template <> struct is_tag<floating_point> : std::true_type {};
	struct unsigned_int {};		 template <> struct is_tag<unsigned_int> : std::true_type {};
	struct signed_int {};		 template <> struct is_tag<signed_int> : std::true_type {};
	struct boolean {};			 template <> struct is_tag<boolean> : std::true_type {};
	struct null {};				 template <> struct is_tag<null> : std::true_type {};

	struct document {};

	template <class T> struct tag { using type = typename T::tag; };
	template <> struct tag<uint64_t> { using type = unsigned_int; };
	template <> struct tag<int64_t> { using type = signed_int; };
	template <> struct tag<uint32_t> { using type = unsigned_int; };
	template <> struct tag<int32_t> { using type = signed_int; };
	template <> struct tag<bool> { using type = boolean; };
	template <> struct tag<std::nullptr_t> { using type = null; };
	template <> struct tag<double> { using type = floating_point; };
	template <class T> using tag_t = typename tag<T>::type;

	template <class T> constexpr auto get_tag(T&&) { return tag_t<std::decay_t<T>>{}; }
	template <class T, class Tag> struct has_tag : std::is_same<tag_t<T>, Tag> {};

	template <class tag, class... T> struct contains_tag {};
	template <class tag, bool head_has_tag, class... T> struct contains_tag_helper {};
	template <class tag, class... T> struct contains_tag_helper<tag, true, T...> { enum { value = true }; };
	template <class tag> struct contains_tag_helper<tag, false> { enum { value = false }; };
	template <class tag, class Head, class... Tail> struct contains_tag_helper<tag, false, Head, Tail...> { enum { value = contains_tag<tag, Tail...>::value }; };
	template <class tag> struct contains_tag<tag> { enum { value = false }; };
	template <class tag, class Head, class... Tail> struct contains_tag<tag, Head, Tail...>
	{
		enum { value = contains_tag_helper<tag, has_tag<Head, tag>::value, Head, Tail...>::value };
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
