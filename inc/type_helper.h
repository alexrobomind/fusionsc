#pragma once

namespace gold_fish
{
	constexpr bool conjunction() { return true; }
	template <class... T> constexpr bool conjunction(bool x, T... tail) { return x && conjunction(tail...); }

	template <class T> constexpr T largest(T x) { return x; }
	template <class T> constexpr T largest(T x, T y) { return x > y ? x : y; }
	template <class Head, class... Tail> constexpr Head largest(Head x, Tail... y) { return largest(x, largest(y...)); }
}
