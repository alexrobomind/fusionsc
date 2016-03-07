#include <gold_fish/uncaught_exception.h>
#include "unit_test.h"

namespace gold_fish
{
	struct Test
	{
		Test(bool e)
			: expected(e)
		{}
		~Test() noexcept(false)
		{
			test(checker() == expected);
		}

		bool expected;
		uncaught_exception_checker checker;
	};

	TEST_CASE(uncaught_no_exceptions)
	{
		test(uncaught_exception_checker()() == false);
	}
	TEST_CASE(uncaught_no_exceptions_destructor)
	{
		{
			Test test(false);
		}
	}
	TEST_CASE(uncaught_exceptions_in_flight)
	{
		try
		{
			Test test(true);
			throw 1;
		}
		catch (int)
		{
		}
	}

	struct Test2
	{
		~Test2()
		{
			try
			{
				Test test(true);
				throw 1;
			}
			catch (int)
			{
			}
		}
	};
	TEST_CASE(try_catch_in_destructor)
	{
		{
			Test2 t;
		}
	}
	TEST_CASE(try_catch_in_destructor_with_uncaught_exception)
	{
		try
		{
			Test2 t;
			throw 1;
		}
		catch (int)
		{
		}
	}

	struct Test3
	{
		~Test3()
		{
			Test test(false);
		}
	};
	TEST_CASE(exception_in_flight_but_local_to_destructor)
	{
		try
		{
			Test3 t;
			throw 1;
		}
		catch (int)
		{
		}
	}
}