#pragma once

#include "CppUnitTest.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#define TEST_CASE(name) \
	TEST_CLASS(C##name) \
	{ public: \
		TEST_METHOD(name); \
	}; \
	void C##name::name()

inline void test(bool x)
{
	if (!x) 
		throw 0;
}

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