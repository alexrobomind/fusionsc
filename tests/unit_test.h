#pragma once

#include <vector>

namespace gold_fish
{
	struct test_case
	{
		const char* name;
		void(*function)();
	};
	std::vector<test_case>& test_cases();

	#define CONCAT2(a,b) a##b
	#define CONCAT(a,b) CONCAT2(a, b)

	#define TEST_CASE(name) \
		void name(); \
		static bool CONCAT(add, __COUNTER__) = []{ gold_fish::test_cases().push_back({#name, &name}); return true; }(); \
		void name()

	#define STRINGIFY(x) #x
	#define TOSTRING(x) STRINGIFY(x)
	#define TEST(...) \
		if (!(__VA_ARGS__)) throw std::exception("failed at " __FILE__ "(" TOSTRING(__LINE__) ")")

	template <class Exception, class Lambda> void expect_exception(Lambda&& l)
	{
		try
		{
			l();
			throw "Exception not thrown";
		}
		catch (const Exception&)
		{
		}
	}
}