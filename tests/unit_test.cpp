#include "unit_test.h"

namespace gold_fish
{
	std::vector<test_case>& test_cases()
	{
		static std::vector<test_case> instance;
		return instance;
	}
}