#include <goldfish/optional.h>
#include "unit_test.h"

namespace goldfish
{
	struct with_invalid_state
	{
		int x;
		struct invalid_state
		{
			static void set(std::aligned_storage_t<sizeof(int), alignof(int)>& data)
			{
				reinterpret_cast<int&>(data) = -1;
			}
		};
	};
	TEST_CASE(default_constructor)
	{
		test(optional<int>() == nullopt);
		test(optional<std::string>() == nullopt);
	}
}