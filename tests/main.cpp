#include "unit_test.h"
#include <iostream>

using namespace std;
using namespace gold_fish;

int main()
{
	int cFailures = 0;

	for (auto&& test : test_cases())
	{
		try
		{
			test.function();
		}
		catch (exception& exc)
		{
			++cFailures;
			cout << test.name << ": " << exc.what() << "\n";
		}
		catch (...)
		{
			++cFailures;
			cout << test.name << " threw an exception\n";
		}
	}
	cout << cFailures << " failures, " << (test_cases().size() - cFailures) << " tests passed\n";
}