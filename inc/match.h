#pragma once

#include <tuple>
#include <type_traits>

namespace gold_fish
{
	namespace details
	{
		template <class... Lambdas> struct info {};

		template <class T, class... Args> struct is_callable
		{
			struct yes {};
			struct no {};
			template <class U> static yes test(decltype(std::declval<U>()(std::declval<Args>()...))*) { return{}; }
			template <class U> static no test(...) { return{}; }
			enum { value = std::is_same<yes, decltype(test<T>(nullptr))>::value };
		};

		template <class Head, class... Tail> struct info<Head, Tail...>
		{
			template <bool /*is_head_callable*/, class... Args> struct first_callable_helper {};
			template <class... Args> struct first_callable
			{
				enum { value = first_callable_helper<is_callable<Head, Args...>::value, Args...>::value };
			};

			template <class... Args> struct first_callable_helper<true, Args...> { enum { value = 0 }; };
			template <class... Args> struct first_callable_helper<false, Args...> { enum { value = 1 + info<Tail...>::first_callable<Args...>::value }; };
		};

		template <class... Lambdas>
		class first_match_object
		{
		public:
			template <class... Args>
			first_match_object(Args&&... args)
				: m_lambdas(std::forward<Args>(args)...)
			{}

			template <class... Args>
			decltype(auto) operator()(Args&&... args)
			{
				return std::get<info<Lambdas...>::first_callable<Args...>::value>(m_lambdas)(std::forward<Args>(args)...);
			}

		private:
			std::tuple<Lambdas...> m_lambdas;
		};

		struct sink_all { template <class U, class V> void operator()(U&&, V&&) {} };

		template <class... Lambdas>
		class first_match_object_symmetric
		{
		public:
			template <class... Args>
			first_match_object_symmetric(Args&&... args)
				: m_lambdas(std::forward<Args>(args)...)
			{}

			template <class A, class B>
			auto operator()(A&& a, B&& b)
			{
				constexpr auto first_direct = info<Lambdas..., sink_all>::first_callable<A, B>::value;
				constexpr auto first_reversed = info<Lambdas..., sink_all>::first_callable<B, A>::value;
				if (first_direct <= first_reversed)
					return std::get<first_direct>(m_lambdas)(std::forward<A>(a), std::forward<B>(b));
				else
					return std::get<first_reversed>(m_lambdas)(std::forward<B>(b), std::forward<A>(a));
			}

		private:
			std::tuple<Lambdas...> m_lambdas;
		};
	}

	template <class... Lambdas>
	decltype(auto) first_match(Lambdas&&... lambdas)
	{
		return details::first_match_object<std::decay_t<Lambdas>...>(std::forward<Lambdas>(lambdas)...);
	}

	template <class... Lambdas>
	auto first_match_symmetric(Lambdas&&... lambdas)
	{
		return details::first_match_object_symmetric<std::decay_t<Lambdas>...>(std::forward<Lambdas>(lambdas)...);
	}
}