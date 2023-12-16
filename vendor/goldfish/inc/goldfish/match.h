#pragma once

#include <tuple>
#include <type_traits>

namespace goldfish
{
	namespace details
	{
		template <class T, class... Args> struct is_callable
		{
			struct yes {};
			struct no {};
			template <class U> static yes test(decltype(std::declval<U>()(std::declval<Args>()...))*) { return{}; }
			template <class U> static no test(...) { return{}; }
			enum { value = std::is_same<yes, decltype(test<T>(nullptr))>::value };
		};

		template <class... Lambdas> struct lambda_trait {};
		template <class Head, class... Tail> struct lambda_trait<Head, Tail...>
		{
			template <bool /*is_head_callable*/, class... Args> struct first_callable_helper {};
			template <class... Args> struct first_callable_index
			{
				enum { value = first_callable_helper<is_callable<Head, Args...>::value, Args...>::value };
			};

			template <class... Args> struct first_callable_helper<true, Args...> { enum { value = 0 }; };
			template <class... Args> struct first_callable_helper<false, Args...> { enum { value = 1 + lambda_trait<Tail...>::first_callable_index<Args...>::value }; };
		};

		template <class... T> struct best_match_object : T...
		{
			template <class... Args> best_match_object(Args&&... args)
				: T(std::forward<Args>(args))...
			{}
		};
	}

	template <class... Lambdas> auto first_match(Lambdas&&... lambdas)
	{
		return [lambdas = std::make_tuple(std::forward<Lambdas>(lambdas)...)](auto&&... args) -> decltype(auto)
		{
			return std::get<details::lambda_trait<Lambdas...>::first_callable_index<decltype(args)...>::value>(lambdas)(std::forward<decltype(args)>(args)...);
		}; 
	}

	template <class... Lambdas> auto best_match(Lambdas&&... lambdas)
	{
		return details::best_match_object<Lambdas...>(std::forward<Lambdas>(lambdas)...);
	}
}